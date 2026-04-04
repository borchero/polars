use std::num::NonZeroUsize;
use std::sync::Arc;

use polars_error::PolarsResult;
use polars_io::pl_async;
use polars_io::utils::sync_on_close::SyncOnCloseType;
use polars_plan::dsl::file_provider::FileProviderArgs;
use polars_plan::dsl::sink::SinkedFileInfo;

use crate::async_executor;
use crate::async_primitives::connector;
use crate::nodes::TaskPriority;
use crate::nodes::io_sinks::components::file_provider::FileProvider;
use crate::nodes::io_sinks::components::file_sink::{FileSinkPermit, FileSinkTaskData};
use crate::nodes::io_sinks::components::size::RowCountAndSize;
use crate::nodes::io_sinks::writers::interface::{FileOpenTaskHandle, FileWriterStarter};
use crate::utils::tokio_handle_ext;

#[derive(Clone)]
pub struct PartitionSinkStarter {
    pub file_provider: Arc<FileProvider>,
    pub writer_starter: Arc<dyn FileWriterStarter>,
    pub sync_on_close: SyncOnCloseType,
    pub num_pipelines_per_sink: NonZeroUsize,
}

impl PartitionSinkStarter {
    pub fn start_sink(
        &self,
        file_provider_args: FileProviderArgs,
        start_position: RowCountAndSize,
        file_permit: FileSinkPermit,
    ) -> PolarsResult<FileSinkTaskData> {
        let file_provider = Arc::clone(&self.file_provider);
        let sinked_file_info_list = file_provider.sinked_file_info_list.clone();

        let (path_tx, path_rx) = tokio::sync::oneshot::channel();

        let file_open_task =
            tokio_handle_ext::AbortOnDropHandle(pl_async::get_runtime().spawn(async move {
                let (writeable, resolved_path) =
                    file_provider.open_file(file_provider_args).await?;
                let _ = path_tx.send(resolved_path);
                Ok(writeable)
            }));

        let (morsel_tx, morsel_rx) = connector::connector();

        let writer_handle = self.writer_starter.start_file_writer(
            morsel_rx,
            FileOpenTaskHandle::new(file_open_task, self.sync_on_close),
            self.num_pipelines_per_sink,
        )?;

        let task_handle = async_executor::spawn(TaskPriority::High, async move {
            let file_stats = writer_handle.await?;

            if let Some(sinked_file_info_list) = sinked_file_info_list {
                if let Ok(Some(path)) = path_rx.await {
                    sinked_file_info_list
                        .file_info_list
                        .lock()
                        .push(SinkedFileInfo {
                            path,
                            stats: file_stats,
                        });
                }
            }

            Ok(file_permit)
        });

        Ok(FileSinkTaskData {
            morsel_tx,
            start_position,
            task_handle,
        })
    }
}
