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

//! Column expression

use std::sync::Arc;

use arrow::{
    datatypes::{DataType, Schema},
    record_batch::RecordBatch,
};

use crate::error::{DataFusionError, Result};
use crate::logical_plan::DFSchema;
use crate::physical_plan::{ColumnarValue, PhysicalExpr};
use arrow::datatypes::Field;

/// Represents the column at a given index in a RecordBatch
#[derive(Debug)]
pub struct Column {
    name: String,
    relation: Option<String>,
}

impl Column {
    /// Create a new column expression
    pub fn new(name: &str) -> Self {
        Self {
            name: name.to_owned(),
            relation: None,
        }
    }

    /// Get the column name
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Create a new column expression with alias
    pub fn new_with_alias(name: &str, relation: Option<String>) -> Self {
        Self {
            name: name.to_owned(),
            relation,
        }
    }

    /// Try to search with prefix and then without
    pub fn lookup_field<'a>(&self, schema: &'a Schema) -> Result<&'a Field> {
        schema.field_with_name(&self.full_name()).or_else(|e| {
            schema
                .fields()
                .iter()
                .find(|f| f.name().ends_with(&format!(".{}", self.name)))
                .ok_or(DataFusionError::ArrowError(e))
        })
    }

    /// Return fully qualified name if alias provided and short name if it's None
    pub fn full_name(&self) -> String {
        format!(
            "{}{}",
            self.relation
                .as_ref()
                .map(|a| format!("{}.", a))
                .unwrap_or_else(|| "".to_string()),
            self.name
        )
    }
}

impl std::fmt::Display for Column {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "{}", self.name)
    }
}

impl PhysicalExpr for Column {
    /// Return a reference to Any that can be used for downcasting
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    /// Get the data type of this expression, given the schema of the input
    fn data_type(&self, input_schema: &DFSchema) -> Result<DataType> {
        Ok(input_schema
            .field_with_name(self.relation.as_deref(), &self.name)?
            .data_type()
            .clone())
    }

    /// Decide whehter this expression is nullable, given the schema of the input
    fn nullable(&self, input_schema: &DFSchema) -> Result<bool> {
        Ok(input_schema
            .field_with_name(self.relation.as_deref(), &self.name)?
            .is_nullable())
    }

    /// Evaluate the expression
    fn evaluate(&self, batch: &RecordBatch) -> Result<ColumnarValue> {
        Ok(ColumnarValue::Array(
            batch
                .column(
                    batch
                        .schema()
                        .index_of(&self.lookup_field(&batch.schema())?.name())?,
                )
                .clone(),
        ))
    }
}

/// Create a column expression
pub fn col(name: &str) -> Arc<dyn PhysicalExpr> {
    Arc::new(Column::new(name))
}
