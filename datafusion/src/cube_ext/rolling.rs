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

use crate::cube_ext::datetime::date_addsub_scalar;
use crate::cube_ext::stream::StreamWithSchema;
use crate::cube_ext::util::{cmp_same_types, lexcmp_array_rows};
use crate::error::DataFusionError;
use crate::execution::context::ExecutionContextState;
use crate::logical_plan::window_frames::WindowFrameBound;
use crate::logical_plan::{
    Column, DFSchemaRef, Expr, LogicalPlan, UserDefinedLogicalNode,
};
use crate::physical_plan::coalesce_batches::concat_batches;
use crate::physical_plan::expressions::PhysicalSortExpr;
use crate::physical_plan::group_scalar::GroupByScalar;
use crate::physical_plan::hash_aggregate::{
    append_value, create_accumulators, create_builder, create_group_by_value,
};
use crate::physical_plan::planner::ExtensionPlanner;
use crate::physical_plan::sort::SortExec;
use crate::physical_plan::{
    collect, AggregateExpr, ColumnarValue, Distribution, ExecutionPlan, Partitioning,
    PhysicalExpr, PhysicalPlanner, SendableRecordBatchStream,
};
use crate::scalar::ScalarValue;
use arrow::array::{make_array, ArrayRef, BooleanBuilder, MutableArrayData, UInt64Array};
use arrow::compute::filter;
use arrow::datatypes::{DataType, Schema, SchemaRef};
use arrow::record_batch::RecordBatch;
use async_trait::async_trait;
use chrono::{TimeZone, Utc};
use hashbrown::HashMap;
use itertools::Itertools;
use sqlparser::ast::RollingOffset;
use std::any::Any;
use std::cmp::{max, Ordering};
use std::convert::TryFrom;
use std::sync::Arc;

#[derive(Debug)]
pub struct RollingWindowAggregate {
    pub schema: DFSchemaRef,
    pub input: LogicalPlan,
    pub dimension: Column,
    pub from: Expr,
    pub to: Expr,
    pub every: Expr,
    pub partition_by: Vec<Column>,
    pub rolling_aggs: Vec<Expr>,
    pub group_by_dimension: Option<Expr>,
    pub aggs: Vec<Expr>,
}

impl UserDefinedLogicalNode for RollingWindowAggregate {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn inputs(&self) -> Vec<&LogicalPlan> {
        vec![&self.input]
    }

    fn schema(&self) -> &DFSchemaRef {
        &self.schema
    }

    fn expressions(&self) -> Vec<Expr> {
        let mut e = vec![
            Expr::Column(self.dimension.clone()),
            self.from.clone(),
            self.to.clone(),
            self.every.clone(),
        ];
        e.extend(self.partition_by.iter().map(|c| Expr::Column(c.clone())));
        e.extend_from_slice(self.rolling_aggs.as_slice());
        e.extend_from_slice(self.aggs.as_slice());
        if let Some(d) = &self.group_by_dimension {
            e.push(d.clone());
        }
        e
    }

    fn fmt_for_explain(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(
            f,
            "ROLLING WINDOW: dimension={}, from={:?}, to={:?}, every={:?}",
            self.dimension, self.from, self.to, self.every
        )
    }

    fn from_template(
        &self,
        exprs: &[Expr],
        inputs: &[LogicalPlan],
    ) -> Arc<dyn UserDefinedLogicalNode + Send + Sync> {
        assert_eq!(inputs.len(), 1);
        assert_eq!(
            exprs.len(),
            4 + self.partition_by.len()
                + self.rolling_aggs.len()
                + self.aggs.len()
                + self.group_by_dimension.as_ref().map(|_| 1).unwrap_or(0)
        );
        let input = inputs[0].clone();
        let dimension = match &exprs[0] {
            Expr::Column(c) => c.clone(),
            o => panic!("Expected column for dimension, got {:?}", o),
        };
        let from = exprs[1].clone();
        let to = exprs[2].clone();
        let every = exprs[3].clone();
        let exprs = &exprs[4..];
        let partition_by = exprs[..self.partition_by.len()]
            .iter()
            .map(|c| match c {
                Expr::Column(c) => c.clone(),
                o => panic!("Expected column for partition_by, got {:?}", o),
            })
            .collect_vec();
        let exprs = &exprs[self.partition_by.len()..];

        let rolling_aggs = exprs[..self.rolling_aggs.len()].to_vec();
        let exprs = &exprs[self.rolling_aggs.len()..];

        let aggs = exprs[..self.aggs.len()].to_vec();
        let exprs = &exprs[self.aggs.len()..];

        let group_by_dimension = if self.group_by_dimension.is_some() {
            debug_assert_eq!(exprs.len(), 1);
            Some(exprs[0].clone())
        } else {
            debug_assert_eq!(exprs.len(), 0);
            None
        };

        Arc::new(RollingWindowAggregate {
            schema: self.schema.clone(),
            input,
            dimension,
            from,
            to,
            every,
            partition_by,
            rolling_aggs,
            group_by_dimension,
            aggs,
        })
    }
}

pub struct Planner;
impl ExtensionPlanner for Planner {
    fn plan_extension(
        &self,
        planner: &dyn PhysicalPlanner,
        node: &dyn UserDefinedLogicalNode,
        _logical_inputs: &[&LogicalPlan],
        physical_inputs: &[Arc<dyn ExecutionPlan>],
        ctx_state: &ExecutionContextState,
    ) -> Result<Option<Arc<dyn ExecutionPlan>>, DataFusionError> {
        use crate::logical_plan;
        use crate::physical_plan::expressions::Column;

        let node = match node.as_any().downcast_ref::<RollingWindowAggregate>() {
            None => return Ok(None),
            Some(n) => n,
        };
        assert_eq!(physical_inputs.len(), 1);
        let input = &physical_inputs[0];
        let input_dfschema = node.input.schema().as_ref();
        let input_schema = input.schema();

        let phys_col = |c: &logical_plan::Column| -> Result<_, DataFusionError> {
            Ok(Column::new(&c.name, input_dfschema.index_of_column(c)?))
        };
        let dimension = phys_col(&node.dimension)?;
        let dimension_type = input_schema.field(dimension.index()).data_type();

        let empty_batch = RecordBatch::new_empty(Arc::new(Schema::new(vec![])));
        let from = planner.create_physical_expr(
            &node.from,
            input_dfschema,
            &input_schema,
            ctx_state,
        )?;
        let from = expect_non_null_scalar("FROM", from.evaluate(&empty_batch)?)?;

        let to = planner.create_physical_expr(
            &node.to,
            input_dfschema,
            &input_schema,
            ctx_state,
        )?;
        let to = expect_non_null_scalar("TO", to.evaluate(&empty_batch)?)?;

        let every = planner.create_physical_expr(
            &node.every,
            input_dfschema,
            &input_schema,
            ctx_state,
        )?;
        let every = expect_non_null_scalar("EVERY", every.evaluate(&empty_batch)?)?;

        if cmp_same_types(&to, &from, true, true) < Ordering::Equal {
            return Err(DataFusionError::Plan("TO is less than FROM".to_string()));
        }
        if cmp_same_types(&add_dim(&from, &every), &from, true, true) <= Ordering::Equal {
            return Err(DataFusionError::Plan("EVERY must be positive".to_string()));
        }

        let rolling_aggs = node
            .rolling_aggs
            .iter()
            .map(|e| -> Result<_, DataFusionError> {
                match e {
                    Expr::RollingAggregate {
                        agg,
                        start,
                        end,
                        offset,
                    } => {
                        let start = frame_bound_to_diff(start, dimension_type)?;
                        let end = frame_bound_to_diff(end, dimension_type)?;
                        let agg = planner.create_aggregate_expr(
                            agg,
                            input_dfschema,
                            &input_schema,
                            ctx_state,
                        )?;
                        Ok(RollingAgg {
                            agg,
                            lower_bound: start,
                            upper_bound: end,
                            offset_to_end: match offset {
                                RollingOffset::Start => false,
                                RollingOffset::End => true,
                            },
                        })
                    }
                    _ => panic!("expected ROLLING() aggregate, got {:?}", e),
                }
            })
            .collect::<Result<Vec<_>, _>>()?;

        let group_by_dimension = node
            .group_by_dimension
            .as_ref()
            .map(|d| {
                planner.create_physical_expr(d, input_dfschema, &input_schema, ctx_state)
            })
            .transpose()?;
        let aggs = node
            .aggs
            .iter()
            .map(|a| {
                planner.create_aggregate_expr(a, input_dfschema, &input_schema, ctx_state)
            })
            .collect::<Result<_, _>>()?;

        // TODO: filter inputs by date.
        // Do preliminary sorting.
        let mut sort_key = Vec::with_capacity(input_schema.fields().len());
        let mut group_key = Vec::with_capacity(input_schema.fields().len() - 1);
        for c in &node.partition_by {
            let c = phys_col(c)?;
            sort_key.push(PhysicalSortExpr {
                expr: Arc::new(c.clone()),
                options: Default::default(),
            });
            group_key.push(c);
        }
        sort_key.push(PhysicalSortExpr {
            expr: Arc::new(dimension.clone()),
            options: Default::default(),
        });

        let sort = Arc::new(SortExec::try_new(sort_key, input.clone())?);

        let schema = node.schema.to_schema_ref();

        Ok(Some(Arc::new(RollingWindowAggExec {
            schema,
            sorted_input: sort,
            group_key,
            rolling_aggs,
            dimension,
            group_by_dimension,
            aggs,
            from,
            to,
            every,
        })))
    }
}

fn frame_bound_to_diff(
    b: &WindowFrameBound,
    dimension: &DataType,
) -> Result<Option<ScalarValue>, DataFusionError> {
    match b {
        WindowFrameBound::CurrentRow => match dimension {
            DataType::Int64 => Ok(Some(ScalarValue::Int64(Some(0)))),
            DataType::Timestamp(_, _) => Ok(Some(ScalarValue::IntervalDayTime(Some(0)))),
            _ => Err(DataFusionError::Plan(format!(
                "unsupported type for window frame bound {}",
                dimension
            ))),
        },
        // Planner checks UNBOUNDED PRECEDING/FOLLOWING are not used for end/start bound.
        WindowFrameBound::Preceding(None) => Ok(None),
        WindowFrameBound::Following(None) => Ok(None),

        WindowFrameBound::Following(Some(v)) => Ok(Some(v.clone())),
        WindowFrameBound::Preceding(Some(v)) => {
            let mut v = v.clone();
            // Note this is probably the only place that can produce negative intervals!
            match &mut v {
                ScalarValue::Int64(Some(i)) => *i = -*i,
                ScalarValue::IntervalYearMonth(Some(i)) => *i = -*i,
                ScalarValue::IntervalDayTime(Some(i)) => *i = -*i,
                v => {
                    return Err(DataFusionError::Internal(format!(
                        "unexpected window frame bound value {}",
                        v
                    )))
                }
            }
            Ok(Some(v))
        }
    }
}

#[derive(Debug, Clone)]
pub struct RollingAgg {
    /// The bound is inclusive.
    pub lower_bound: Option<ScalarValue>,
    /// The bound is inclusive.
    pub upper_bound: Option<ScalarValue>,
    pub agg: Arc<dyn AggregateExpr>,
    /// When true, all calculations must be done for the last point in the interval.
    pub offset_to_end: bool,
}

#[derive(Debug)]
pub struct RollingWindowAggExec {
    pub schema: SchemaRef,
    pub sorted_input: Arc<dyn ExecutionPlan>,
    pub group_key: Vec<crate::physical_plan::expressions::Column>,
    pub rolling_aggs: Vec<RollingAgg>,
    pub dimension: crate::physical_plan::expressions::Column,
    pub group_by_dimension: Option<Arc<dyn PhysicalExpr>>,
    pub aggs: Vec<Arc<dyn AggregateExpr>>,
    pub from: ScalarValue,
    pub to: ScalarValue,
    pub every: ScalarValue,
}

#[async_trait]
impl ExecutionPlan for RollingWindowAggExec {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn schema(&self) -> SchemaRef {
        self.schema.clone()
    }

    fn output_partitioning(&self) -> Partitioning {
        Partitioning::UnknownPartitioning(1)
    }

    fn required_child_distribution(&self) -> Distribution {
        Distribution::UnspecifiedDistribution
    }

    fn children(&self) -> Vec<Arc<dyn ExecutionPlan>> {
        vec![self.sorted_input.clone()]
    }

    fn with_new_children(
        &self,
        mut children: Vec<Arc<dyn ExecutionPlan>>,
    ) -> Result<Arc<dyn ExecutionPlan>, DataFusionError> {
        assert_eq!(children.len(), 1);
        Ok(Arc::new(RollingWindowAggExec {
            schema: self.schema(),
            sorted_input: children.remove(0),
            group_key: self.group_key.clone(),
            rolling_aggs: self.rolling_aggs.clone(),
            dimension: self.dimension.clone(),
            group_by_dimension: self.group_by_dimension.clone(),
            aggs: self.aggs.clone(),
            from: self.from.clone(),
            to: self.to.clone(),
            every: self.every.clone(),
        }))
    }

    #[tracing::instrument(level = "trace", skip(self))]
    async fn execute(
        &self,
        partition: usize,
    ) -> Result<SendableRecordBatchStream, DataFusionError> {
        assert_eq!(partition, 0);
        // Sort keeps everything in-memory anyway. So don't stream and keep implementation simple.
        let batches = collect(self.sorted_input.clone()).await?;
        let num_rows = batches.iter().map(|b| b.num_rows()).sum();
        let input = concat_batches(&self.sorted_input.schema(), &batches, num_rows)?;

        let num_rows = input.num_rows();
        let key_cols = self
            .group_key
            .iter()
            .map(|c| input.columns()[c.index()].clone())
            .collect_vec();

        let other_cols = input
            .columns()
            .iter()
            .enumerate()
            .filter_map(|(i, c)| {
                if self.dimension.index() == i
                    || self.group_key.iter().any(|c| c.index() == i)
                {
                    None
                } else {
                    Some(c.clone())
                }
            })
            .collect_vec();
        let agg_inputs = self
            .rolling_aggs
            .iter()
            .map(|r| compute_agg_inputs(r.agg.as_ref(), &input))
            .collect::<Result<Vec<_>, _>>()?;
        let mut accumulators = self
            .rolling_aggs
            .iter()
            .map(|r| r.agg.create_accumulator())
            .collect::<Result<Vec<_>, _>>()?;
        let mut dimension = input.column(self.dimension.index()).clone();
        let dim_iter_type = self.from.get_datatype();
        if dimension.data_type() != &dim_iter_type {
            // This is to upcast timestamps to nanosecond precision.
            dimension = arrow::compute::cast(&dimension, &dim_iter_type)?;
        }

        let extra_aggs_dimension = self
            .group_by_dimension
            .as_ref()
            .map(|d| -> Result<_, DataFusionError> {
                let mut d = d.evaluate(&input)?.into_array(num_rows);
                if d.data_type() != &dim_iter_type {
                    // This is to upcast timestamps to nanosecond precision.
                    d = arrow::compute::cast(&d, &dim_iter_type)?;
                }
                Ok(d)
            })
            .transpose()?;
        let extra_aggs_inputs = self
            .aggs
            .iter()
            .map(|a| compute_agg_inputs(a.as_ref(), &input))
            .collect::<Result<Vec<_>, _>>()?;

        let mut out_dim = create_builder(&self.from);
        let mut out_keys = key_cols
            .iter()
            .map(|c| MutableArrayData::new(vec![c.data()], true, 0))
            .collect_vec();
        let mut out_aggs = Vec::with_capacity(self.rolling_aggs.len());
        // This filter must be applied prior to returning the values.
        let mut out_aggs_keep = BooleanBuilder::new(0);
        let extra_agg_nulls = self
            .aggs
            .iter()
            .map(|a| ScalarValue::try_from(a.field()?.data_type()))
            .collect::<Result<Vec<_>, _>>()?;
        let mut out_extra_aggs = extra_agg_nulls.iter().map(create_builder).collect_vec();
        let mut out_other = other_cols
            .iter()
            .map(|c| MutableArrayData::new(vec![c.data()], true, 0))
            .collect_vec();
        let mut row_i = 0;
        let mut any_group_had_values = vec![];
        while row_i < num_rows {
            let group_start = row_i;
            while row_i + 1 < num_rows
                && lexcmp_array_rows(key_cols.iter(), row_i, row_i + 1).is_eq()
            {
                row_i += 1;
            }
            let group_end = row_i + 1;
            row_i = group_end;

            // Compute aggregate on each interesting date and add them to the output.
            let mut had_values = Vec::new();
            for (ri, r) in self.rolling_aggs.iter().enumerate() {
                // Avoid running indefinitely due to all kinds of errors.
                let mut window_start = group_start;
                let mut window_end = group_start;
                let offset_to_end = if r.offset_to_end {
                    Some(&self.every)
                } else {
                    None
                };

                let mut d = self.from.clone();
                let mut d_iter = 0;
                while cmp_same_types(&d, &self.to, true, true) <= Ordering::Equal {
                    while window_start < group_end
                        && !meets_lower_bound(
                            &ScalarValue::try_from_array(&dimension, window_start)
                                .unwrap(),
                            &d,
                            r.lower_bound.as_ref(),
                            offset_to_end,
                        )
                    {
                        window_start += 1;
                    }
                    window_end = max(window_end, window_start);
                    while window_end < group_end
                        && meets_upper_bound(
                            &ScalarValue::try_from_array(&dimension, window_end).unwrap(),
                            &d,
                            r.upper_bound.as_ref(),
                            offset_to_end,
                        )
                    {
                        window_end += 1;
                    }
                    if had_values.len() == d_iter {
                        had_values.push(window_start != window_end);
                    } else {
                        had_values[d_iter] |= window_start != window_end;
                    }

                    // TODO: pick easy performance wins for SUM() and AVG() with subtraction.
                    //       Also experiment with interval trees for other accumulators.
                    accumulators[ri].reset();
                    accumulators[ri].update_batch(
                        &agg_inputs[ri]
                            .iter()
                            .map(|a| a.slice(window_start, window_end - window_start))
                            .collect_vec(),
                    )?;
                    let v = accumulators[ri].evaluate()?;
                    if ri == out_aggs.len() {
                        out_aggs.push(create_builder(&v));
                    }
                    append_value(out_aggs[ri].as_mut(), &v)?;

                    const MAX_DIM_ITERATIONS: usize = 10_000_000;
                    d_iter += 1;
                    if d_iter == MAX_DIM_ITERATIONS {
                        return Err(DataFusionError::Execution(
                            "reached the limit of iterations for rolling window dimensions"
                                .to_string(),
                        ));
                    }
                    d = add_dim(&d, &self.every);
                }
            }

            if any_group_had_values.is_empty() {
                any_group_had_values = had_values.clone();
            } else {
                for i in 0..had_values.len() {
                    any_group_had_values[i] |= had_values[i];
                }
            }

            // Compute non-rolling aggregates for the group.
            let mut dim_to_extra_aggs = HashMap::new();
            if let Some(key) = &extra_aggs_dimension {
                let mut key_to_rows = HashMap::new();
                for i in group_start..group_end {
                    let key = create_group_by_value(key, i)?;
                    key_to_rows.entry(key).or_insert(Vec::new()).push(i as u64);
                }

                for (k, rows) in key_to_rows {
                    let mut accumulators = create_accumulators(&self.aggs)?;
                    let rows = UInt64Array::from(rows);
                    let mut values = Vec::with_capacity(accumulators.len());
                    for i in 0..accumulators.len() {
                        let accum_inputs = extra_aggs_inputs[i]
                            .iter()
                            .map(|a| arrow::compute::take(a.as_ref(), &rows, None))
                            .collect::<Result<Vec<_>, _>>()?;
                        accumulators[i].update_batch(&accum_inputs)?;
                        values.push(accumulators[i].evaluate()?);
                    }

                    dim_to_extra_aggs.insert(k, values);
                }
            }

            // Add keys, dimension and non-aggregate columns to the output.
            let mut d = self.from.clone();
            let mut d_iter = 0;
            let mut matching_row_lower_bound = 0;
            while cmp_same_types(&d, &self.to, true, true) <= Ordering::Equal {
                if !had_values[d_iter] {
                    out_aggs_keep.append_value(false)?;

                    d_iter += 1;
                    d = add_dim(&d, &self.every);
                    continue;
                } else {
                    out_aggs_keep.append_value(true)?;
                }
                append_value(out_dim.as_mut(), &d)?;
                for i in 0..key_cols.len() {
                    out_keys[i].extend(0, group_start, group_start + 1)
                }
                // Add aggregates.
                match dim_to_extra_aggs.get(&GroupByScalar::try_from(&d)?) {
                    Some(aggs) => {
                        for i in 0..out_extra_aggs.len() {
                            append_value(out_extra_aggs[i].as_mut(), &aggs[i])?
                        }
                    }
                    None => {
                        for i in 0..out_extra_aggs.len() {
                            append_value(out_extra_aggs[i].as_mut(), &extra_agg_nulls[i])?
                        }
                    }
                }
                // Find the matching row to add other columns.
                while matching_row_lower_bound < group_end
                    && cmp_same_types(
                        &ScalarValue::try_from_array(
                            &dimension,
                            matching_row_lower_bound,
                        )
                        .unwrap(),
                        &d,
                        true,
                        true,
                    ) < Ordering::Equal
                {
                    matching_row_lower_bound += 1;
                }
                if matching_row_lower_bound < group_end
                    && ScalarValue::try_from_array(&dimension, matching_row_lower_bound)
                        .unwrap()
                        == d
                {
                    for i in 0..other_cols.len() {
                        out_other[i].extend(
                            0,
                            matching_row_lower_bound,
                            matching_row_lower_bound + 1,
                        );
                    }
                } else {
                    for o in &mut out_other {
                        o.extend_nulls(1);
                    }
                }
                d_iter += 1;
                d = add_dim(&d, &self.every);
            }
        }

        // We also promise to produce null values for dates missing in the input.
        let mut d = self.from.clone();
        let mut num_empty_dims = 0;
        for i in 0..any_group_had_values.len() {
            if !any_group_had_values[i] {
                append_value(out_dim.as_mut(), &d)?;
                num_empty_dims += 1;
            }
            d = add_dim(&d, &self.every);
        }
        for c in &mut out_keys {
            c.extend_nulls(num_empty_dims);
        }
        for c in &mut out_other {
            c.extend_nulls(num_empty_dims);
        }
        for i in 0..out_aggs.len() {
            accumulators[i].reset();
            let null = accumulators[i].evaluate()?;
            for _ in 0..num_empty_dims {
                append_value(out_aggs[i].as_mut(), &null)?;
            }
        }
        for i in 0..out_extra_aggs.len() {
            let null = &extra_agg_nulls[i];
            for _ in 0..num_empty_dims {
                append_value(out_extra_aggs[i].as_mut(), &null)?;
            }
        }
        for _ in 0..num_empty_dims {
            out_aggs_keep.append_value(true)?;
        }

        // Produce final output.
        if out_dim.is_empty() {
            return Ok(Box::pin(StreamWithSchema::wrap(
                self.schema(),
                futures::stream::empty(),
            )));
        };

        let mut r =
            Vec::with_capacity(1 + out_keys.len() + out_other.len() + out_aggs.len());
        r.push(out_dim.finish());
        for k in out_keys {
            r.push(make_array(k.freeze()));
        }
        for o in out_other {
            r.push(make_array(o.freeze()));
        }

        let out_aggs_keep = out_aggs_keep.finish();
        for mut a in out_aggs {
            r.push(filter(a.finish().as_ref(), &out_aggs_keep)?);
        }

        for mut a in out_extra_aggs {
            r.push(a.finish())
        }

        let r = RecordBatch::try_new(self.schema(), r)?;
        Ok(Box::pin(StreamWithSchema::wrap(
            self.schema(),
            futures::stream::iter(vec![Ok(r)]),
        )))
    }
}

fn add_dim(l: &ScalarValue, r: &ScalarValue) -> ScalarValue {
    match (l, r) {
        (ScalarValue::Int64(Some(l)), ScalarValue::Int64(Some(r))) => {
            ScalarValue::Int64(Some(l + r))
        }
        (
            ScalarValue::TimestampNanosecond(Some(l)),
            i @ (ScalarValue::IntervalDayTime(Some(_))
            | ScalarValue::IntervalYearMonth(Some(_))),
        ) => {
            let v = date_addsub_scalar(Utc.timestamp_nanos(*l), i.clone(), true).unwrap();
            ScalarValue::TimestampNanosecond(Some(v.timestamp_nanos()))
        }
        _ => panic!("unsupported dimension type"),
    }
}

fn compute_agg_inputs(
    a: &dyn AggregateExpr,
    input: &RecordBatch,
) -> Result<Vec<ArrayRef>, DataFusionError> {
    a.expressions()
        .iter()
        .map(|e| -> Result<_, DataFusionError> {
            Ok(e.evaluate(input)?.into_array(input.num_rows()))
        })
        .collect()
}

/// Returns `(value, current+bounds)` pair that can be used for comparison to check window bounds.
fn prepare_bound_compare(
    value: &ScalarValue,
    current: &ScalarValue,
    bound: &ScalarValue,
    offset_to_end: Option<&ScalarValue>,
) -> (i64, i64) {
    let mut added = add_dim(current, bound);
    if let Some(offset) = offset_to_end {
        added = add_dim(&added, offset)
    }

    let (mut added, value) = match (added, value) {
        (ScalarValue::Int64(Some(a)), ScalarValue::Int64(Some(v))) => (a, v),
        (
            ScalarValue::TimestampNanosecond(Some(a)),
            ScalarValue::TimestampNanosecond(Some(v)),
        ) => (a, v),
        (a, v) => panic!("unsupported values in rolling window: ({:?}, {:?})", a, v),
    };

    if offset_to_end.is_some() {
        added -= 1
    }
    (*value, added)
}

fn meets_lower_bound(
    value: &ScalarValue,
    current: &ScalarValue,
    bound: Option<&ScalarValue>,
    offset_to_end: Option<&ScalarValue>,
) -> bool {
    let bound = match bound {
        Some(p) => p,
        None => return true,
    };
    assert!(!bound.is_null());
    assert!(!current.is_null());
    if value.is_null() {
        return false;
    }
    let (value, added) = prepare_bound_compare(value, current, bound, offset_to_end);
    added <= value
}

fn meets_upper_bound(
    value: &ScalarValue,
    current: &ScalarValue,
    bound: Option<&ScalarValue>,
    offset_to_end: Option<&ScalarValue>,
) -> bool {
    let bound = match bound {
        Some(p) => p,
        None => return true,
    };
    assert!(!bound.is_null());
    assert!(!current.is_null());
    if value.is_null() {
        return false;
    }
    let (value, added) = prepare_bound_compare(value, current, bound, offset_to_end);
    value <= added
}

fn expect_non_null_scalar(
    var: &str,
    v: ColumnarValue,
) -> Result<ScalarValue, DataFusionError> {
    match v {
        ColumnarValue::Array(_) => {
            return Err(DataFusionError::Plan(format!(
                "expected scalar for {}, got array",
                var
            )))
        }
        ColumnarValue::Scalar(s) if s.is_null() => {
            return Err(DataFusionError::Plan(format!("{} must not be null", var)))
        }
        ColumnarValue::Scalar(s) => return Ok(s),
    }
}
