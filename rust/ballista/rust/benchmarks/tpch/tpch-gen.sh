#!/bin/bash
# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

BALLISTA_VERSION=0.4.2-SNAPSHOT

#set -e

docker build -t ballistacompute/ballista-tpchgen:$BALLISTA_VERSION -f tpchgen.dockerfile .

# Generate data into the ./data directory if it does not already exist
FILE=./data/supplier.tbl
if test -f "$FILE"; then
    echo "$FILE exists."
else
  mkdir data 2>/dev/null
  docker run -v `pwd`/data:/data -it --rm ballistacompute/ballista-tpchgen:$BALLISTA_VERSION
  ls -l data
fi