use std::sync::Arc;

use arrow::datatypes::ArrowSchemaRef;
use polars_buffer::Buffer;
use polars_error::PolarsResult;
use polars_io::parquet::write::BatchedWriter;
use polars_io::prelude::{FileMetadata, KeyValueMetadata};
use polars_parquet::parquet::error::ParquetResult;
use polars_parquet::parquet::metadata::ThriftFileMetadata;
use polars_parquet::parquet::write::reduce_statistics;
use polars_parquet::write::{Encoding, FileWriter, SchemaDescriptor, WriteOptions};
use polars_plan::dsl::sink::{SinkedFileColumnStats, SinkedFileStats};

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

        let mut parquet_writer = BatchedWriter::new(
            std::sync::Mutex::new(FileWriter::new_with_parquet_schema(
                &mut *buffered_file,
                Arc::unwrap_or_clone(arrow_schema),
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
        let file_stats = build_file_stats(metadata, file_size, metadata_size)?;

        drop(buffered_file);

        file.close(sync_on_close)?;

        Ok(file_stats)
    }
}

fn build_file_stats(
    metadata: ThriftFileMetadata,
    file_size: u64,
    metadata_size: u64,
) -> PolarsResult<SinkedFileStats> {
    let file_metadata = FileMetadata::try_from_thrift(metadata)?;
    let num_columns = file_metadata.schema().columns().len();

    let column_stats = (0..num_columns)
        .map(|col_idx| build_column_stats(&file_metadata, col_idx))
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
    col_idx: usize,
) -> PolarsResult<SinkedFileColumnStats> {
    // Get basic information
    let column = &file_metadata.schema().columns()[col_idx];
    let compressed_size_bytes = file_metadata
        .row_groups
        .iter()
        .map(|rg| rg.parquet_columns()[col_idx].compressed_size())
        .sum();

    // Obtain stats across row groups
    let all_stats = file_metadata
        .row_groups
        .iter()
        .filter_map(|rg| {
            rg.parquet_columns()[col_idx]
                .statistics()
                .map(|r| r.map(Some))
        })
        .collect::<ParquetResult<Vec<_>>>()?;
    let all_stats_opt = all_stats.iter().collect::<Vec<_>>();
    let full_stats = reduce_statistics(&all_stats_opt)?;

    // Build result
    let stats = SinkedFileColumnStats {
        name: column.path_in_schema.clone(),
        compressed_size_bytes,
        null_count: full_stats.as_ref().and_then(|s| s.null_count()),
        min_value: full_stats.as_ref().and_then(|s| s.serialize().min_value),
        max_value: full_stats.as_ref().and_then(|s| s.serialize().max_value),
    };
    Ok(stats)
}
