use std::sync::Arc;

use arrow::datatypes::{ArrowDataType, ArrowSchema, ArrowSchemaRef, Field};
use polars_buffer::Buffer;
use polars_core::prelude::{DataType, Series};
use polars_error::PolarsResult;
use polars_io::parquet::write::BatchedWriter;
use polars_io::prelude::{FileMetadata, KeyValueMetadata};
use polars_parquet::parquet::metadata::ThriftFileMetadata;
use polars_parquet::read::statistics::deserialize_all;
use polars_parquet::write::{Encoding, FileWriter, SchemaDescriptor, WriteOptions};
use polars_plan::dsl::sink::{SinkedFileColumnStats, SinkedFileStats};
use polars_utils::IdxSize;
use polars_utils::pl_str::PlSmallStr;

use crate::async_executor;
use crate::nodes::io_sinks::writers::interface::FileOpenTaskHandle;
use crate::nodes::io_sinks::writers::parquet::EncodedRowGroup;

pub struct IOWriter {
    pub file: FileOpenTaskHandle,
    pub encoded_row_group_rx: tokio::sync::mpsc::Receiver<
        async_executor::AbortOnDropHandle<PolarsResult<EncodedRowGroup>>,
    >,
    pub arrow_schema: ArrowSchemaRef,
    pub schema_descriptor: Arc<SchemaDescriptor>,
    pub write_options: WriteOptions,
    pub encodings: Buffer<Vec<Encoding>>,
    pub key_value_metadata: Option<KeyValueMetadata>,
    pub num_leaf_columns: usize,
}

impl IOWriter {
    pub async fn run(self) -> PolarsResult<SinkedFileStats> {
        let IOWriter {
            file,
            mut encoded_row_group_rx,
            arrow_schema,
            schema_descriptor,
            write_options,
            encodings,
            key_value_metadata,
            num_leaf_columns,
        } = self;

        let (mut file, sync_on_close) = file.await?;
        let mut buffered_file = file.as_buffered();

        let arrow_schema = Arc::unwrap_or_clone(arrow_schema);
        let mut parquet_writer = BatchedWriter::new(
            std::sync::Mutex::new(FileWriter::new_with_parquet_schema(
                &mut *buffered_file,
                arrow_schema.clone(),
                Arc::unwrap_or_clone(schema_descriptor),
                write_options,
            )),
            encodings,
            write_options,
            false,
            key_value_metadata,
        );

        while let Some(handle) = encoded_row_group_rx.recv().await {
            let EncodedRowGroup {
                num_rows,
                data,
                morsel_permit,
            } = handle.await?;
            assert_eq!(data.len(), num_leaf_columns);
            parquet_writer.write_row_group(num_rows as u64, &data)?;
            drop(data);
            drop(morsel_permit);
        }

        let (file_size, metadata_size) = parquet_writer.finish()?;
        let metadata = parquet_writer.into_metadata();
        let file_stats = build_file_stats(arrow_schema, metadata, file_size, metadata_size)?;

        drop(buffered_file);

        file.close(sync_on_close)?;

        Ok(file_stats)
    }
}

fn build_file_stats(
    arrow_schema: ArrowSchema,
    metadata: ThriftFileMetadata,
    file_size: u64,
    metadata_size: u64,
) -> PolarsResult<SinkedFileStats> {
    let file_metadata = FileMetadata::try_from_thrift(metadata)?;
    let num_columns = file_metadata.schema().columns().len();

    // Flatten the arrow schema fields to leaf fields to match parquet's leaf column
    // layout. This is correct because both `to_parquet_type` (which builds the parquet
    // schema from the arrow schema) and `SchemaDescriptor::new` (which collects leaf
    // columns) use the same DFS order over struct children.
    let mut leaf_fields = Vec::with_capacity(num_columns);
    for field in arrow_schema.iter_values() {
        collect_leaf_fields(field, &mut leaf_fields);
    }
    let column_stats = (0..num_columns)
        .map(|col_idx| build_column_stats(&file_metadata, leaf_fields[col_idx], col_idx))
        .collect::<PolarsResult<Vec<_>>>()?;

    let stats = SinkedFileStats {
        num_rows: file_metadata.num_rows as u64,
        file_size_bytes: file_size,
        footer_size_bytes: metadata_size,
        columns: column_stats,
    };
    Ok(stats)
}

fn build_column_stats(
    file_metadata: &FileMetadata,
    field: &Field,
    col_idx: usize,
) -> PolarsResult<SinkedFileColumnStats> {
    // Get basic information
    let column = &file_metadata.schema().columns()[col_idx];
    let compressed_size_bytes: i64 = file_metadata
        .row_groups
        .iter()
        .map(|rg| rg.parquet_columns()[col_idx].compressed_size())
        .sum();

    // Deserialize statistics across all row groups using deserialize_all
    let (null_count, min_value, max_value) = if !file_metadata.row_groups.is_empty() {
        match deserialize_all(field, &file_metadata.row_groups, col_idx)? {
            Some(stats) => {
                let null_count: IdxSize = stats.null_count.non_null_values_iter().sum();

                let dtype = DataType::from_arrow_dtype(field.dtype());
                let min_series =
                    Series::from_chunk_and_dtype(PlSmallStr::EMPTY, stats.min_value, &dtype)?;
                let min_value = min_series.min_reduce().ok().map(|s| s.value().clone());

                let max_series =
                    Series::from_chunk_and_dtype(PlSmallStr::EMPTY, stats.max_value, &dtype)?;
                let max_value = max_series.max_reduce().ok().map(|s| s.value().clone());

                (Some(null_count), min_value, max_value)
            },
            None => (None, None, None),
        }
    } else {
        (None, None, None)
    };

    // Build result
    let stats = SinkedFileColumnStats {
        name: column.path_in_schema.clone(),
        compressed_size_bytes: compressed_size_bytes as u64,
        null_count,
        min_value,
        max_value,
    };
    Ok(stats)
}

/// Collect leaf-level arrow fields in DFS order, matching the parquet leaf column layout.
fn collect_leaf_fields<'a>(field: &'a Field, out: &mut Vec<&'a Field>) {
    match field.dtype() {
        ArrowDataType::Struct(children) => {
            for child in children {
                collect_leaf_fields(child, out);
            }
        },
        ArrowDataType::List(inner)
        | ArrowDataType::LargeList(inner)
        | ArrowDataType::FixedSizeList(inner, _) => {
            collect_leaf_fields(inner, out);
        },
        ArrowDataType::Map(inner, _) => {
            collect_leaf_fields(inner, out);
        },
        _ => out.push(field),
    }
}
