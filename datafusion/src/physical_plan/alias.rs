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

//! Adds an alias to the schemas of input batches

use crate::error::Result;
use crate::logical_plan::DFSchemaRef;
use crate::physical_plan::{
    ExecutionPlan, OptimizerHints, Partitioning, RecordBatchStream,
    SendableRecordBatchStream,
};
use arrow::datatypes::SchemaRef;
use arrow::error::Result as ArrowResult;
use arrow::record_batch::RecordBatch;
use async_trait::async_trait;
use futures::{Stream, StreamExt};
use std::any::Any;
use std::pin::Pin;
use std::sync::Arc;
use std::task::{Context, Poll};

/// Prefix schema with alias schema for every batch in stream
#[derive(Debug)]
pub struct AliasedSchemaExec {
    alias: String,
    input: Arc<dyn ExecutionPlan>,
}

impl AliasedSchemaExec {
    /// Wrap with AliasedSchema
    pub fn wrap(
        alias: Option<String>,
        input: Arc<dyn ExecutionPlan>,
    ) -> Arc<dyn ExecutionPlan> {
        if let Some(alias) = alias {
            Arc::new(Self { alias, input })
        } else {
            input
        }
    }
}

#[async_trait]
impl ExecutionPlan for AliasedSchemaExec {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn schema(&self) -> DFSchemaRef {
        let schema = self.input.schema();
        Arc::new(schema.alias(Some(self.alias.as_str())).unwrap())
    }

    fn output_partitioning(&self) -> Partitioning {
        self.input.output_partitioning()
    }

    fn children(&self) -> Vec<Arc<dyn ExecutionPlan>> {
        vec![self.input.clone()]
    }

    fn with_new_children(
        &self,
        children: Vec<Arc<dyn ExecutionPlan>>,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        Ok(Arc::new(Self {
            input: children[0].clone(),
            alias: self.alias.clone(),
        }))
    }

    async fn execute(&self, partition: usize) -> Result<SendableRecordBatchStream> {
        Ok(Box::pin(AliasedSchemaStream {
            input: self.input.execute(partition).await?,
            schema: self.schema().to_schema_ref(),
        }))
    }

    fn output_hints(&self) -> OptimizerHints {
        self.input.output_hints()
    }
}

/// Alias Schema for every batch
pub struct AliasedSchemaStream {
    input: SendableRecordBatchStream,
    schema: SchemaRef,
}

impl Stream for AliasedSchemaStream {
    type Item = ArrowResult<RecordBatch>;

    fn poll_next(
        mut self: Pin<&mut Self>,
        cx: &mut Context<'_>,
    ) -> Poll<Option<Self::Item>> {
        self.input
            .poll_next_unpin(cx)
            .map(|option| -> Option<ArrowResult<RecordBatch>> {
                option.map(|batch| {
                    RecordBatch::try_new(self.schema.clone(), batch?.columns().to_vec())
                })
            })
    }
}

impl RecordBatchStream for AliasedSchemaStream {
    fn schema(&self) -> SchemaRef {
        self.schema.clone()
    }
}
