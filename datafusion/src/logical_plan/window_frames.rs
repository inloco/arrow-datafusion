// Licensed to the Apache Software Foundation (ASF) under one
// or more contributor license agreements.  See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership.  The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License.  You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing,
// software distributed under the License is distributed on an
// "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, either express or implied.  See the License for the
// specific language governing permissions and limitations
// under the License.

//! Window frame
//!
//! The frame-spec determines which output rows are read by an aggregate window function. The frame-spec consists of four parts:
//! - A frame type - either ROWS, RANGE or GROUPS,
//! - A starting frame boundary,
//! - An ending frame boundary,
//! - An EXCLUDE clause.

use crate::error::{DataFusionError, Result};
use crate::execution::context::ExecutionContextState;
use crate::logical_plan::Expr;
use crate::scalar::ScalarValue;
use crate::sql::planner::SqlToRel;
use serde_derive::{Deserialize, Serialize};
use sqlparser::ast;
use sqlparser::ast::DateTimeField;
use std::cmp::Ordering;
use std::convert::TryInto;
use std::convert::{From, TryFrom};
use std::fmt;

/// The frame-spec determines which output rows are read by an aggregate window function.
///
/// The ending frame boundary can be omitted (if the BETWEEN and AND keywords that surround the
/// starting frame boundary are also omitted), in which case the ending frame boundary defaults to
/// CURRENT ROW.
#[derive(Debug, Clone, PartialEq)]
pub struct WindowFrame {
    /// A frame type - either ROWS, RANGE or GROUPS
    pub units: WindowFrameUnits,
    /// A starting frame boundary
    pub start_bound: WindowFrameBound,
    /// An ending frame boundary
    pub end_bound: WindowFrameBound,
}

impl fmt::Display for WindowFrame {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(
            f,
            "{} BETWEEN {} AND {}",
            self.units, self.start_bound, self.end_bound
        )?;
        Ok(())
    }
}

impl TryFrom<ast::WindowFrame> for WindowFrame {
    type Error = DataFusionError;

    fn try_from(value: ast::WindowFrame) -> Result<Self> {
        let start_bound = value.start_bound.try_into()?;
        let end_bound = value
            .end_bound
            .map(WindowFrameBound::try_from)
            .unwrap_or(Ok(WindowFrameBound::CurrentRow))?;
        check_window_bound_order(&start_bound, &end_bound)?;

        let is_allowed_range_bound = |s: &ScalarValue| match s {
            ScalarValue::Int64(Some(i)) => *i == 0,
            _ => false,
        };

        let units = value.units.into();
        if units == WindowFrameUnits::Range {
            for bound in &[&start_bound, &end_bound] {
                match bound {
                    WindowFrameBound::Preceding(Some(v))
                    | WindowFrameBound::Following(Some(v)) if !is_allowed_range_bound(v) => {
                        Err(DataFusionError::NotImplemented(format!(
                            "With WindowFrameUnits={}, the bound cannot be {} PRECEDING or FOLLOWING at the moment",
                            units, v
                        )))
                    }
                    _ => Ok(()),
                }?;
            }
        }
        Ok(Self {
            units,
            start_bound,
            end_bound,
        })
    }
}

#[allow(missing_docs)]
pub fn check_window_bound_order(
    start_bound: &WindowFrameBound,
    end_bound: &WindowFrameBound,
) -> Result<()> {
    if let WindowFrameBound::Following(None) = start_bound {
        Err(DataFusionError::Execution(
            "Invalid window frame: start bound cannot be unbounded following".to_owned(),
        ))
    } else if let WindowFrameBound::Preceding(None) = end_bound {
        Err(DataFusionError::Execution(
            "Invalid window frame: end bound cannot be unbounded preceding".to_owned(),
        ))
    } else {
        match start_bound.logical_cmp(&end_bound)  {
            None =>  Err(DataFusionError::Execution(format!(
                    "Invalid window frame: start bound ({}) is incompatble with the end bound ({})",
                    start_bound, end_bound
                ))),
            Some(o) if o > Ordering::Equal => Err(DataFusionError::Execution(format!(
                    "Invalid window frame: start bound ({}) cannot be larger than end bound ({})",
                    start_bound, end_bound
                ))),
            Some(_) => Ok(()),
        }
    }
}

impl Default for WindowFrame {
    fn default() -> Self {
        WindowFrame {
            units: WindowFrameUnits::Range,
            start_bound: WindowFrameBound::Preceding(None),
            end_bound: WindowFrameBound::CurrentRow,
        }
    }
}

/// There are five ways to describe starting and ending frame boundaries:
///
/// 1. UNBOUNDED PRECEDING
/// 2. <expr> PRECEDING
/// 3. CURRENT ROW
/// 4. <expr> FOLLOWING
/// 5. UNBOUNDED FOLLOWING
///
/// in this implementation we'll only allow <expr> to be u64 (i.e. no dynamic boundary)
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum WindowFrameBound {
    /// 1. UNBOUNDED PRECEDING
    /// The frame boundary is the first row in the partition.
    ///
    /// 2. <expr> PRECEDING
    /// <expr> must be a non-negative constant numeric expression. The boundary is a row that
    /// is <expr> "units" prior to the current row.
    Preceding(Option<ScalarValue>),
    /// 3. The current row.
    ///
    /// For RANGE and GROUPS frame types, peers of the current row are also
    /// included in the frame, unless specifically excluded by the EXCLUDE clause.
    /// This is true regardless of whether CURRENT ROW is used as the starting or ending frame
    /// boundary.
    CurrentRow,
    /// 4. This is the same as "<expr> PRECEDING" except that the boundary is <expr> units after the
    /// current rather than before the current row.
    ///
    /// 5. UNBOUNDED FOLLOWING
    /// The frame boundary is the last row in the partition.
    Following(Option<ScalarValue>),
}

impl TryFrom<ast::WindowFrameBound> for WindowFrameBound {
    type Error = DataFusionError;

    fn try_from(value: ast::WindowFrameBound) -> Result<Self> {
        let value_to_scalar = |v| -> Result<_> {
            match v {
                None => Ok(None),
                Some(ast::Value::Number(v, _)) => match v.parse() {
                        Err(_) => Err(DataFusionError::Plan(format!("could not convert window frame bound '{}' to int64", v))),
                        Ok(v) => Ok(Some(ScalarValue::Int64(Some(v)))),
                },
                Some(ast::Value::Interval { value, leading_field, leading_precision, last_field, fractional_seconds_precision })
                    => Ok(Some(interval_to_scalar(&value, &leading_field, &leading_precision, &last_field, &fractional_seconds_precision)?)),
                Some(o) => Err(DataFusionError::Plan(format!("window frame bound must be a positive integer or an INTERVAL, got {}", o))),
            }
        };

        match value {
            ast::WindowFrameBound::Preceding(v) => {
                Ok(Self::Preceding(value_to_scalar(v)?))
            }
            ast::WindowFrameBound::Following(v) => {
                Ok(Self::Following(value_to_scalar(v)?))
            }
            ast::WindowFrameBound::CurrentRow => Ok(Self::CurrentRow),
        }
    }
}

fn interval_to_scalar(
    value: &str,
    leading_field: &Option<DateTimeField>,
    leading_precision: &Option<u64>,
    last_field: &Option<DateTimeField>,
    fractional_seconds_precision: &Option<u64>,
) -> Result<ScalarValue> {
    match SqlToRel::<ExecutionContextState>::sql_interval_to_literal(
        value,
        leading_field,
        leading_precision,
        last_field,
        fractional_seconds_precision,
    )? {
        Expr::Literal(v) => Ok(v),
        o => panic!("unexpected result of interval_to_literal: {:?}", o),
    }
}

impl fmt::Display for WindowFrameBound {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            WindowFrameBound::CurrentRow => f.write_str("CURRENT ROW"),
            WindowFrameBound::Preceding(None) => f.write_str("UNBOUNDED PRECEDING"),
            WindowFrameBound::Following(None) => f.write_str("UNBOUNDED FOLLOWING"),
            WindowFrameBound::Preceding(Some(n)) => write!(f, "{} PRECEDING", n),
            WindowFrameBound::Following(Some(n)) => write!(f, "{} FOLLOWING", n),
        }
    }
}

impl WindowFrameBound {
    /// We deliberately avoid implementing [PartialCmp] as this is non-structural comparison.
    /// The reason is that we severly limit a combination of scalars accepted by this function.
    pub fn logical_cmp(&self, other: &Self) -> Option<Ordering> {
        use WindowFrameBound::{CurrentRow, Following, Preceding};
        let ord = |v: &WindowFrameBound| match v {
            Preceding(_) => 0,
            CurrentRow => 1,
            Following(_) => 2,
        };

        let lo = ord(self);
        let ro = ord(other);
        let o = lo.cmp(&ro);
        if o != Ordering::Equal {
            return Some(o);
        }

        let (l, r) = match (self, other) {
            (Preceding(Some(l)), Preceding(Some(r))) => (r, l), // reverse comparison order.
            (Following(Some(l)), Following(Some(r))) => (l, r),

            (CurrentRow, CurrentRow) => return Some(Ordering::Equal),

            (Preceding(None), Preceding(None)) => return Some(Ordering::Equal),
            (Preceding(None), Preceding(Some(_))) => return Some(Ordering::Less),
            (Preceding(Some(_)), Preceding(None)) => return Some(Ordering::Greater),

            (Following(None), Following(None)) => return Some(Ordering::Equal),
            (Following(Some(_)), Following(None)) => return Some(Ordering::Less),
            (Following(None), Following(Some(_))) => return Some(Ordering::Greater),
            _ => panic!("unhandled bounds: {} and {}", self, other),
        };

        match (l, r) {
            (ScalarValue::Int64(Some(l)), ScalarValue::Int64(Some(r))) => Some(l.cmp(r)),
            (
                ScalarValue::IntervalDayTime(Some(l)),
                ScalarValue::IntervalDayTime(Some(r)),
            ) => Some(l.cmp(r)),
            (
                ScalarValue::IntervalYearMonth(Some(l)),
                ScalarValue::IntervalYearMonth(Some(r)),
            ) => Some(l.cmp(r)),
            // Cannot compare other types.
            _ => None,
        }
    }
}

/// There are three frame types: ROWS, GROUPS, and RANGE. The frame type determines how the
/// starting and ending boundaries of the frame are measured.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WindowFrameUnits {
    /// The ROWS frame type means that the starting and ending boundaries for the frame are
    /// determined by counting individual rows relative to the current row.
    Rows,
    /// The RANGE frame type requires that the ORDER BY clause of the window have exactly one
    /// term. Call that term "X". With the RANGE frame type, the elements of the frame are
    /// determined by computing the value of expression X for all rows in the partition and framing
    /// those rows for which the value of X is within a certain range of the value of X for the
    /// current row.
    Range,
    /// The GROUPS frame type means that the starting and ending boundaries are determine
    /// by counting "groups" relative to the current group. A "group" is a set of rows that all have
    /// equivalent values for all all terms of the window ORDER BY clause.
    Groups,
}

impl fmt::Display for WindowFrameUnits {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.write_str(match self {
            WindowFrameUnits::Rows => "ROWS",
            WindowFrameUnits::Range => "RANGE",
            WindowFrameUnits::Groups => "GROUPS",
        })
    }
}

impl From<ast::WindowFrameUnits> for WindowFrameUnits {
    fn from(value: ast::WindowFrameUnits) -> Self {
        match value {
            ast::WindowFrameUnits::Range => Self::Range,
            ast::WindowFrameUnits::Groups => Self::Groups,
            ast::WindowFrameUnits::Rows => Self::Rows,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_window_frame_creation() -> Result<()> {
        let window_frame = ast::WindowFrame {
            units: ast::WindowFrameUnits::Range,
            start_bound: ast::WindowFrameBound::Following(None),
            end_bound: None,
        };
        let result = WindowFrame::try_from(window_frame);
        assert_eq!(
            result.err().unwrap().to_string(),
            "Execution error: Invalid window frame: start bound cannot be unbounded following".to_owned()
        );

        let window_frame = ast::WindowFrame {
            units: ast::WindowFrameUnits::Range,
            start_bound: ast::WindowFrameBound::Preceding(None),
            end_bound: Some(ast::WindowFrameBound::Preceding(None)),
        };
        let result = WindowFrame::try_from(window_frame);
        assert_eq!(
            result.err().unwrap().to_string(),
            "Execution error: Invalid window frame: end bound cannot be unbounded preceding".to_owned()
        );

        let window_frame = ast::WindowFrame {
            units: ast::WindowFrameUnits::Range,
            start_bound: ast::WindowFrameBound::Preceding(Some(1)),
            end_bound: Some(ast::WindowFrameBound::Preceding(Some(2))),
        };
        let result = WindowFrame::try_from(window_frame);
        assert_eq!(
            result.err().unwrap().to_string(),
            "Execution error: Invalid window frame: start bound (1 PRECEDING) cannot be larger than end bound (2 PRECEDING)".to_owned()
        );

        let window_frame = ast::WindowFrame {
            units: ast::WindowFrameUnits::Range,
            start_bound: ast::WindowFrameBound::Preceding(Some(2)),
            end_bound: Some(ast::WindowFrameBound::Preceding(Some(1))),
        };
        let result = WindowFrame::try_from(window_frame);
        assert_eq!(
            result.err().unwrap().to_string(),
            "This feature is not implemented: With WindowFrameUnits=RANGE, the bound cannot be 2 PRECEDING or FOLLOWING at the moment".to_owned()
        );

        let window_frame = ast::WindowFrame {
            units: ast::WindowFrameUnits::Rows,
            start_bound: ast::WindowFrameBound::Preceding(Some(2)),
            end_bound: Some(ast::WindowFrameBound::Preceding(Some(1))),
        };
        let result = WindowFrame::try_from(window_frame);
        assert!(result.is_ok());
        Ok(())
    }

    #[test]
    fn test_eq() {
        assert_eq!(
            WindowFrameBound::Preceding(Some(0)),
            WindowFrameBound::CurrentRow
        );
        assert_eq!(
            WindowFrameBound::CurrentRow,
            WindowFrameBound::Following(Some(0))
        );
        assert_eq!(
            WindowFrameBound::Following(Some(2)),
            WindowFrameBound::Following(Some(2))
        );
        assert_eq!(
            WindowFrameBound::Following(None),
            WindowFrameBound::Following(None)
        );
        assert_eq!(
            WindowFrameBound::Preceding(Some(2)),
            WindowFrameBound::Preceding(Some(2))
        );
        assert_eq!(
            WindowFrameBound::Preceding(None),
            WindowFrameBound::Preceding(None)
        );
    }

    #[test]
    fn test_ord() {
        assert!(WindowFrameBound::Preceding(Some(1)) < WindowFrameBound::CurrentRow);
        // ! yes this is correct!
        assert!(
            WindowFrameBound::Preceding(Some(2)) < WindowFrameBound::Preceding(Some(1))
        );
        assert!(
            WindowFrameBound::Preceding(Some(u64::MAX))
                < WindowFrameBound::Preceding(Some(u64::MAX - 1))
        );
        assert!(
            WindowFrameBound::Preceding(None)
                < WindowFrameBound::Preceding(Some(1000000))
        );
        assert!(
            WindowFrameBound::Preceding(None)
                < WindowFrameBound::Preceding(Some(u64::MAX))
        );
        assert!(WindowFrameBound::Preceding(None) < WindowFrameBound::Following(Some(0)));
        assert!(
            WindowFrameBound::Preceding(Some(1)) < WindowFrameBound::Following(Some(1))
        );
        assert!(WindowFrameBound::CurrentRow < WindowFrameBound::Following(Some(1)));
        assert!(
            WindowFrameBound::Following(Some(1)) < WindowFrameBound::Following(Some(2))
        );
        assert!(WindowFrameBound::Following(Some(2)) < WindowFrameBound::Following(None));
        assert!(
            WindowFrameBound::Following(Some(u64::MAX))
                < WindowFrameBound::Following(None)
        );
    }
}
