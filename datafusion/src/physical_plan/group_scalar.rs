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

//! Defines scalars used to construct groups, ex. in GROUP BY clauses.

use std::convert::{From, TryFrom};

use crate::cube_ext::ordfloat::{OrdF32, OrdF64};
use crate::error::{DataFusionError, Result};
use crate::scalar::ScalarValue;
use arrow::datatypes::DataType;

/// Enumeration of types that can be used in a GROUP BY expression
#[allow(missing_docs)]
#[derive(Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Clone)]
pub enum GroupByScalar {
    Null,
    Float32(OrdF32),
    Float64(OrdF64),
    UInt8(u8),
    UInt16(u16),
    UInt32(u32),
    UInt64(u64),
    Int8(i8),
    Int16(i16),
    Int32(i32),
    Int64(i64),
    Int96(i128),
    // TODO
    Utf8(String),
    LargeUtf8(String),
    Boolean(bool),
    TimeMillisecond(i64),
    TimeMicrosecond(i64),
    TimeNanosecond(i64),
    Int64Decimal(i64, u8),
    Int96Decimal(i128, u8),
    Date32(i32),
}

impl TryFrom<&ScalarValue> for GroupByScalar {
    type Error = DataFusionError;

    fn try_from(scalar_value: &ScalarValue) -> Result<Self> {
        Ok(match scalar_value {
            ScalarValue::Float32(Some(v)) => GroupByScalar::Float32(OrdF32::from(*v)),
            ScalarValue::Float64(Some(v)) => GroupByScalar::Float64(OrdF64::from(*v)),
            ScalarValue::Boolean(Some(v)) => GroupByScalar::Boolean(*v),
            ScalarValue::Int8(Some(v)) => GroupByScalar::Int8(*v),
            ScalarValue::Int16(Some(v)) => GroupByScalar::Int16(*v),
            ScalarValue::Int32(Some(v)) => GroupByScalar::Int32(*v),
            ScalarValue::Int64(Some(v)) => GroupByScalar::Int64(*v),
            ScalarValue::Int96(Some(v)) => GroupByScalar::Int96(*v),
            // TODO
            ScalarValue::UInt8(Some(v)) => GroupByScalar::UInt8(*v),
            ScalarValue::UInt16(Some(v)) => GroupByScalar::UInt16(*v),
            ScalarValue::UInt32(Some(v)) => GroupByScalar::UInt32(*v),
            ScalarValue::UInt64(Some(v)) => GroupByScalar::UInt64(*v),
            ScalarValue::Int64Decimal(Some(v), size) => {
                GroupByScalar::Int64Decimal(*v, *size)
            }
            ScalarValue::Int96Decimal(Some(v), size) => {
                GroupByScalar::Int96Decimal(*v, *size)
            }
            ScalarValue::TimestampMillisecond(Some(v)) => {
                GroupByScalar::TimeMillisecond(*v)
            }
            ScalarValue::TimestampMicrosecond(Some(v)) => {
                GroupByScalar::TimeMicrosecond(*v)
            }
            ScalarValue::TimestampNanosecond(Some(v)) => {
                GroupByScalar::TimeNanosecond(*v)
            }
            ScalarValue::Utf8(Some(v)) => GroupByScalar::Utf8(v.clone()),
            ScalarValue::LargeUtf8(Some(v)) => GroupByScalar::LargeUtf8(v.clone()),
            ScalarValue::Float32(None)
            | ScalarValue::Float64(None)
            | ScalarValue::Boolean(None)
            | ScalarValue::Int8(None)
            | ScalarValue::Int16(None)
            | ScalarValue::Int32(None)
            | ScalarValue::Int64(None)
            | ScalarValue::Int96(None)
            | ScalarValue::UInt8(None)
            | ScalarValue::UInt16(None)
            | ScalarValue::UInt32(None)
            | ScalarValue::UInt64(None)
            | ScalarValue::Utf8(None)
            | ScalarValue::Int64Decimal(None, _)
            | ScalarValue::Int96Decimal(None, _)
            | ScalarValue::TimestampMillisecond(None)
            | ScalarValue::TimestampMicrosecond(None)
            | ScalarValue::TimestampNanosecond(None) => GroupByScalar::Null,
            v => {
                return Err(DataFusionError::Internal(format!(
                    "Cannot convert a ScalarValue with associated DataType {:?}",
                    v.get_datatype()
                )))
            }
        })
    }
}

impl GroupByScalar {
    /// Convert to ScalarValue.
    pub fn to_scalar(&self, ty: &DataType) -> ScalarValue {
        let r = match self {
            GroupByScalar::Null => {
                ScalarValue::try_from(ty).expect("could not create null")
            }
            GroupByScalar::Float32(v) => ScalarValue::Float32(Some((*v).0)),
            GroupByScalar::Float64(v) => ScalarValue::Float64(Some((*v).0)),
            GroupByScalar::Boolean(v) => ScalarValue::Boolean(Some(*v)),
            GroupByScalar::Int8(v) => ScalarValue::Int8(Some(*v)),
            GroupByScalar::Int16(v) => ScalarValue::Int16(Some(*v)),
            GroupByScalar::Int32(v) => ScalarValue::Int32(Some(*v)),
            GroupByScalar::Int64(v) => ScalarValue::Int64(Some(*v)),
            GroupByScalar::Int96(v) => ScalarValue::Int96(Some(*v)),
            // TODO
            GroupByScalar::UInt8(v) => ScalarValue::UInt8(Some(*v)),
            GroupByScalar::UInt16(v) => ScalarValue::UInt16(Some(*v)),
            GroupByScalar::UInt32(v) => ScalarValue::UInt32(Some(*v)),
            GroupByScalar::UInt64(v) => ScalarValue::UInt64(Some(*v)),
            GroupByScalar::Utf8(v) => ScalarValue::Utf8(Some(v.to_string())),
            GroupByScalar::LargeUtf8(v) => ScalarValue::LargeUtf8(Some(v.to_string())),
            GroupByScalar::Int64Decimal(v, size) => {
                ScalarValue::Int64Decimal(Some(*v), *size)
            }
            GroupByScalar::Int96Decimal(v, size) => {
                ScalarValue::Int96Decimal(Some(*v), *size)
            }
            GroupByScalar::TimeMillisecond(v) => {
                ScalarValue::TimestampMillisecond(Some(*v))
            }
            GroupByScalar::TimeMicrosecond(v) => {
                ScalarValue::TimestampMicrosecond(Some(*v))
            }
            GroupByScalar::TimeNanosecond(v) => {
                ScalarValue::TimestampNanosecond(Some(*v))
            }
            GroupByScalar::Date32(v) => ScalarValue::Date32(Some(*v)),
        };
        debug_assert_eq!(&r.get_datatype(), ty);
        r
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::error::DataFusionError;

    macro_rules! scalar_eq_test {
        ($TYPE:expr, $VALUE:expr) => {{
            let scalar_value = $TYPE($VALUE);
            let a = GroupByScalar::try_from(&scalar_value).unwrap();

            let scalar_value = $TYPE($VALUE);
            let b = GroupByScalar::try_from(&scalar_value).unwrap();

            assert_eq!(a, b);
        }};
    }

    #[test]
    fn test_scalar_ne_non_std() {
        // Test only Scalars with non native Eq, Hash
        scalar_eq_test!(ScalarValue::Float32, Some(1.0));
        scalar_eq_test!(ScalarValue::Float64, Some(1.0));
    }

    macro_rules! scalar_ne_test {
        ($TYPE:expr, $LVALUE:expr, $RVALUE:expr) => {{
            let scalar_value = $TYPE($LVALUE);
            let a = GroupByScalar::try_from(&scalar_value).unwrap();

            let scalar_value = $TYPE($RVALUE);
            let b = GroupByScalar::try_from(&scalar_value).unwrap();

            assert_ne!(a, b);
        }};
    }

    #[test]
    fn test_scalar_eq_non_std() {
        // Test only Scalars with non native Eq, Hash
        scalar_ne_test!(ScalarValue::Float32, Some(1.0), Some(2.0));
        scalar_ne_test!(ScalarValue::Float64, Some(1.0), Some(2.0));
    }

    #[test]
    fn from_scalar_holding_none() {
        let scalar_value = ScalarValue::Int8(None);
        let result = GroupByScalar::try_from(&scalar_value);
        assert_eq!(result.unwrap(), GroupByScalar::Null);
    }

    #[test]
    fn from_scalar_unsupported() {
        // Use any ScalarValue type not supported by GroupByScalar.
        let scalar_value = ScalarValue::Binary(Some(vec![1, 2]));
        let result = GroupByScalar::try_from(&scalar_value);

        match result {
            Err(DataFusionError::Internal(error_message)) => assert_eq!(
                error_message,
                String::from(
                    "Cannot convert a ScalarValue with associated DataType Binary"
                )
            ),
            _ => panic!("Unexpected result"),
        }
    }

    #[test]
    fn size_of_group_by_scalar() {
        assert_eq!(std::mem::size_of::<GroupByScalar>(), 32);
    }
}
