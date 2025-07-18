use std::hash::BuildHasher;

use arrow::bitmap::MutableBitmap;
use either::Either;
use polars::prelude::*;
use polars_ffi::version_0::SeriesExport;
#[cfg(feature = "pivot")]
use polars_lazy::frame::pivot::{pivot, pivot_stable};
use pyo3::IntoPyObjectExt;
use pyo3::exceptions::PyIndexError;
use pyo3::prelude::*;
use pyo3::pybacked::PyBackedStr;
use pyo3::types::{PyList, PyType};

use self::row_encode::{_get_rows_encoded_ca, _get_rows_encoded_ca_unordered};
use super::PyDataFrame;
use crate::conversion::Wrap;
use crate::error::PyPolarsErr;
use crate::map::dataframe::{
    apply_lambda_unknown, apply_lambda_with_bool_out_type, apply_lambda_with_primitive_out_type,
    apply_lambda_with_string_out_type,
};
use crate::prelude::strings_to_pl_smallstr;
use crate::py_modules::polars;
use crate::series::{PySeries, ToPySeries, ToSeries};
use crate::utils::EnterPolarsExt;
use crate::{PyExpr, PyLazyFrame};

#[pymethods]
impl PyDataFrame {
    #[new]
    pub fn __init__(columns: Vec<PySeries>) -> PyResult<Self> {
        let columns = columns.to_series();
        // @scalar-opt
        let columns = columns.into_iter().map(|s| s.into()).collect();
        let df = DataFrame::new(columns).map_err(PyPolarsErr::from)?;
        Ok(PyDataFrame::new(df))
    }

    pub fn estimated_size(&self) -> usize {
        self.df.estimated_size()
    }

    pub fn dtype_strings(&self) -> Vec<String> {
        self.df
            .get_columns()
            .iter()
            .map(|s| format!("{}", s.dtype()))
            .collect()
    }

    pub fn add(&self, py: Python<'_>, s: &PySeries) -> PyResult<Self> {
        py.enter_polars_df(|| &self.df + &s.series)
    }

    pub fn sub(&self, py: Python<'_>, s: &PySeries) -> PyResult<Self> {
        py.enter_polars_df(|| &self.df - &s.series)
    }

    pub fn mul(&self, py: Python<'_>, s: &PySeries) -> PyResult<Self> {
        py.enter_polars_df(|| &self.df * &s.series)
    }

    pub fn div(&self, py: Python<'_>, s: &PySeries) -> PyResult<Self> {
        py.enter_polars_df(|| &self.df / &s.series)
    }

    pub fn rem(&self, py: Python<'_>, s: &PySeries) -> PyResult<Self> {
        py.enter_polars_df(|| &self.df % &s.series)
    }

    pub fn add_df(&self, py: Python<'_>, s: &Self) -> PyResult<Self> {
        py.enter_polars_df(|| &self.df + &s.df)
    }

    pub fn sub_df(&self, py: Python<'_>, s: &Self) -> PyResult<Self> {
        py.enter_polars_df(|| &self.df - &s.df)
    }

    pub fn mul_df(&self, py: Python<'_>, s: &Self) -> PyResult<Self> {
        py.enter_polars_df(|| &self.df * &s.df)
    }

    pub fn div_df(&self, py: Python<'_>, s: &Self) -> PyResult<Self> {
        py.enter_polars_df(|| &self.df / &s.df)
    }

    pub fn rem_df(&self, py: Python<'_>, s: &Self) -> PyResult<Self> {
        py.enter_polars_df(|| &self.df % &s.df)
    }

    #[pyo3(signature = (n, with_replacement, shuffle, seed=None))]
    pub fn sample_n(
        &self,
        py: Python<'_>,
        n: &PySeries,
        with_replacement: bool,
        shuffle: bool,
        seed: Option<u64>,
    ) -> PyResult<Self> {
        py.enter_polars_df(|| self.df.sample_n(&n.series, with_replacement, shuffle, seed))
    }

    #[pyo3(signature = (frac, with_replacement, shuffle, seed=None))]
    pub fn sample_frac(
        &self,
        py: Python<'_>,
        frac: &PySeries,
        with_replacement: bool,
        shuffle: bool,
        seed: Option<u64>,
    ) -> PyResult<Self> {
        py.enter_polars_df(|| {
            self.df
                .sample_frac(&frac.series, with_replacement, shuffle, seed)
        })
    }

    pub fn rechunk(&self, py: Python) -> PyResult<Self> {
        py.enter_polars_df(|| {
            let mut df = self.df.clone();
            df.as_single_chunk_par();
            Ok(df)
        })
    }

    /// Format `DataFrame` as String
    pub fn as_str(&self) -> String {
        format!("{:?}", self.df)
    }

    pub fn get_columns(&self) -> Vec<PySeries> {
        let cols = self.df.get_columns().to_vec();
        cols.to_pyseries()
    }

    /// Get column names
    pub fn columns(&self) -> Vec<&str> {
        self.df.get_column_names_str()
    }

    /// set column names
    pub fn set_column_names(&mut self, names: Vec<PyBackedStr>) -> PyResult<()> {
        self.df
            .set_column_names(names.iter().map(|x| &**x))
            .map_err(PyPolarsErr::from)?;
        Ok(())
    }

    /// Get datatypes
    pub fn dtypes<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyList>> {
        let iter = self
            .df
            .iter()
            .map(|s| Wrap(s.dtype().clone()).into_pyobject(py).unwrap());
        PyList::new(py, iter)
    }

    pub fn n_chunks(&self) -> usize {
        self.df.first_col_n_chunks()
    }

    pub fn shape(&self) -> (usize, usize) {
        self.df.shape()
    }

    pub fn height(&self) -> usize {
        self.df.height()
    }

    pub fn width(&self) -> usize {
        self.df.width()
    }

    pub fn is_empty(&self) -> bool {
        self.df.is_empty()
    }

    pub fn hstack(&self, py: Python<'_>, columns: Vec<PySeries>) -> PyResult<Self> {
        let columns = columns.to_series();
        // @scalar-opt
        let columns = columns.into_iter().map(Into::into).collect::<Vec<_>>();
        py.enter_polars_df(|| self.df.hstack(&columns))
    }

    pub fn hstack_mut(&mut self, py: Python<'_>, columns: Vec<PySeries>) -> PyResult<()> {
        let columns = columns.to_series();
        // @scalar-opt
        let columns = columns.into_iter().map(Into::into).collect::<Vec<_>>();
        py.enter_polars(|| self.df.hstack_mut(&columns))?;
        Ok(())
    }

    pub fn vstack(&self, py: Python<'_>, other: &PyDataFrame) -> PyResult<Self> {
        py.enter_polars_df(|| self.df.vstack(&other.df))
    }

    pub fn vstack_mut(&mut self, py: Python<'_>, other: &PyDataFrame) -> PyResult<()> {
        py.enter_polars(|| self.df.vstack_mut(&other.df))?;
        Ok(())
    }

    pub fn extend(&mut self, py: Python<'_>, other: &PyDataFrame) -> PyResult<()> {
        py.enter_polars(|| self.df.extend(&other.df))?;
        Ok(())
    }

    pub fn drop_in_place(&mut self, name: &str) -> PyResult<PySeries> {
        let s = self.df.drop_in_place(name).map_err(PyPolarsErr::from)?;
        let s = s.take_materialized_series();
        Ok(PySeries { series: s })
    }

    pub fn to_series(&self, index: isize) -> PyResult<PySeries> {
        let df = &self.df;

        let index_adjusted = if index < 0 {
            df.width().checked_sub(index.unsigned_abs())
        } else {
            Some(usize::try_from(index).unwrap())
        };

        let s = index_adjusted.and_then(|i| df.select_at_idx(i));
        match s {
            Some(s) => Ok(PySeries::new(s.as_materialized_series().clone())),
            None => Err(PyIndexError::new_err(
                polars_err!(oob = index, df.width()).to_string(),
            )),
        }
    }

    pub fn get_column_index(&self, name: &str) -> PyResult<usize> {
        Ok(self
            .df
            .try_get_column_index(name)
            .map_err(PyPolarsErr::from)?)
    }

    pub fn get_column(&self, name: &str) -> PyResult<PySeries> {
        let series = self
            .df
            .column(name)
            .map(|s| PySeries::new(s.as_materialized_series().clone()))
            .map_err(PyPolarsErr::from)?;
        Ok(series)
    }

    pub fn select(&self, py: Python<'_>, columns: Vec<PyBackedStr>) -> PyResult<Self> {
        py.enter_polars_df(|| self.df.select(columns.iter().map(|x| &**x)))
    }

    pub fn gather(&self, py: Python<'_>, indices: Wrap<Vec<IdxSize>>) -> PyResult<Self> {
        let indices = indices.0;
        let indices = IdxCa::from_vec("".into(), indices);
        py.enter_polars_df(|| self.df.take(&indices))
    }

    pub fn gather_with_series(&self, py: Python<'_>, indices: &PySeries) -> PyResult<Self> {
        let indices = indices.series.idx().map_err(PyPolarsErr::from)?;
        py.enter_polars_df(|| self.df.take(indices))
    }

    pub fn replace(&mut self, column: &str, new_col: PySeries) -> PyResult<()> {
        self.df
            .replace(column, new_col.series)
            .map_err(PyPolarsErr::from)?;
        Ok(())
    }

    pub fn replace_column(&mut self, index: usize, new_column: PySeries) -> PyResult<()> {
        self.df
            .replace_column(index, new_column.series)
            .map_err(PyPolarsErr::from)?;
        Ok(())
    }

    pub fn insert_column(&mut self, index: usize, column: PySeries) -> PyResult<()> {
        self.df
            .insert_column(index, column.series)
            .map_err(PyPolarsErr::from)?;
        Ok(())
    }

    #[pyo3(signature = (offset, length=None))]
    pub fn slice(&self, py: Python<'_>, offset: i64, length: Option<usize>) -> PyResult<Self> {
        py.enter_polars_df(|| {
            Ok(self
                .df
                .slice(offset, length.unwrap_or_else(|| self.df.height())))
        })
    }

    pub fn head(&self, py: Python<'_>, n: usize) -> PyResult<Self> {
        py.enter_polars_df(|| Ok(self.df.head(Some(n))))
    }

    pub fn tail(&self, py: Python<'_>, n: usize) -> PyResult<Self> {
        py.enter_polars_df(|| Ok(self.df.tail(Some(n))))
    }

    pub fn is_unique(&self, py: Python) -> PyResult<PySeries> {
        py.enter_polars_series(|| self.df.is_unique())
    }

    pub fn is_duplicated(&self, py: Python) -> PyResult<PySeries> {
        py.enter_polars_series(|| self.df.is_duplicated())
    }

    pub fn equals(&self, py: Python<'_>, other: &PyDataFrame, null_equal: bool) -> PyResult<bool> {
        if null_equal {
            py.enter_polars_ok(|| self.df.equals_missing(&other.df))
        } else {
            py.enter_polars_ok(|| self.df.equals(&other.df))
        }
    }

    #[pyo3(signature = (name, offset=None))]
    pub fn with_row_index(
        &self,
        py: Python<'_>,
        name: &str,
        offset: Option<IdxSize>,
    ) -> PyResult<Self> {
        py.enter_polars_df(|| self.df.with_row_index(name.into(), offset))
    }

    pub fn _to_metadata(&self) -> Self {
        Self {
            df: self.df._to_metadata(),
        }
    }

    pub fn group_by_map_groups(
        &self,
        by: Vec<PyBackedStr>,
        lambda: PyObject,
        maintain_order: bool,
    ) -> PyResult<Self> {
        let gb = if maintain_order {
            self.df.group_by_stable(by.iter().map(|x| &**x))
        } else {
            self.df.group_by(by.iter().map(|x| &**x))
        }
        .map_err(PyPolarsErr::from)?;

        let function = move |df: DataFrame| {
            Python::with_gil(|py| {
                let pypolars = polars(py).bind(py);
                let pydf = PyDataFrame::new(df);
                let python_df_wrapper =
                    pypolars.getattr("wrap_df").unwrap().call1((pydf,)).unwrap();

                // Call the lambda and get a python-side DataFrame wrapper.
                let result_df_wrapper = match lambda.call1(py, (python_df_wrapper,)) {
                    Ok(pyobj) => pyobj,
                    Err(e) => panic!("UDF failed: {}", e.value(py)),
                };
                let py_pydf = result_df_wrapper.getattr(py, "_df").expect(
                    "Could not get DataFrame attribute '_df'. Make sure that you return a DataFrame object.",
                );

                let pydf = py_pydf.extract::<PyDataFrame>(py).unwrap();
                Ok(pydf.df)
            })
        };
        // We don't use `py.allow_threads(|| gb.par_apply(..)` because that segfaulted
        // due to code related to Pyo3 or rayon, cannot reproduce it in native polars.
        // So we lose parallelism, but it doesn't really matter because we are GIL bound anyways
        // and this function should not be used in idiomatic polars anyway.
        let df = gb.apply(function).map_err(PyPolarsErr::from)?;

        Ok(df.into())
    }

    #[allow(clippy::should_implement_trait)]
    pub fn clone(&self) -> Self {
        PyDataFrame::new(self.df.clone())
    }

    #[cfg(feature = "pivot")]
    #[pyo3(signature = (on, index, value_name=None, variable_name=None))]
    pub fn unpivot(
        &self,
        py: Python<'_>,
        on: Vec<PyBackedStr>,
        index: Vec<PyBackedStr>,
        value_name: Option<&str>,
        variable_name: Option<&str>,
    ) -> PyResult<Self> {
        use polars_ops::pivot::UnpivotDF;
        let args = UnpivotArgsIR {
            on: strings_to_pl_smallstr(on),
            index: strings_to_pl_smallstr(index),
            value_name: value_name.map(|s| s.into()),
            variable_name: variable_name.map(|s| s.into()),
        };

        py.enter_polars_df(|| self.df.unpivot2(args))
    }

    #[cfg(feature = "pivot")]
    #[pyo3(signature = (on, index, values, maintain_order, sort_columns, aggregate_expr, separator))]
    pub fn pivot_expr(
        &self,
        py: Python<'_>,
        on: Vec<String>,
        index: Option<Vec<String>>,
        values: Option<Vec<String>>,
        maintain_order: bool,
        sort_columns: bool,
        aggregate_expr: Option<PyExpr>,
        separator: Option<&str>,
    ) -> PyResult<Self> {
        let fun = if maintain_order { pivot_stable } else { pivot };
        let agg_expr = aggregate_expr.map(|expr| expr.inner);
        py.enter_polars_df(|| {
            fun(
                &self.df,
                on,
                index,
                values,
                sort_columns,
                agg_expr,
                separator,
            )
        })
    }

    pub fn partition_by(
        &self,
        py: Python<'_>,
        by: Vec<String>,
        maintain_order: bool,
        include_key: bool,
    ) -> PyResult<Vec<Self>> {
        let out = py.enter_polars(|| {
            if maintain_order {
                self.df.partition_by_stable(by, include_key)
            } else {
                self.df.partition_by(by, include_key)
            }
        })?;

        // SAFETY: PyDataFrame is a repr(transparent) DataFrame.
        Ok(unsafe { std::mem::transmute::<Vec<DataFrame>, Vec<PyDataFrame>>(out) })
    }

    pub fn lazy(&self) -> PyLazyFrame {
        self.df.clone().lazy().into()
    }

    #[pyo3(signature = (columns, separator, drop_first=false, drop_nulls=false))]
    pub fn to_dummies(
        &self,
        py: Python<'_>,
        columns: Option<Vec<String>>,
        separator: Option<&str>,
        drop_first: bool,
        drop_nulls: bool,
    ) -> PyResult<Self> {
        py.enter_polars_df(|| match columns {
            Some(cols) => self.df.columns_to_dummies(
                cols.iter().map(|x| x as &str).collect(),
                separator,
                drop_first,
                drop_nulls,
            ),
            None => self.df.to_dummies(separator, drop_first, drop_nulls),
        })
    }

    pub fn null_count(&self, py: Python) -> PyResult<Self> {
        py.enter_polars_df(|| Ok(self.df.null_count()))
    }

    #[pyo3(signature = (lambda, output_type, inference_size))]
    pub fn map_rows(
        &mut self,
        lambda: Bound<PyAny>,
        output_type: Option<Wrap<DataType>>,
        inference_size: usize,
    ) -> PyResult<(PyObject, bool)> {
        Python::with_gil(|py| {
            // needed for series iter
            self.df.as_single_chunk_par();
            let df = &self.df;

            use apply_lambda_with_primitive_out_type as apply;
            #[rustfmt::skip]
            let out = match output_type.map(|dt| dt.0) {
                Some(DataType::Int32) => apply::<Int32Type>(df, py, lambda, 0, None)?.into_series(),
                Some(DataType::Int64) => apply::<Int64Type>(df, py, lambda, 0, None)?.into_series(),
                Some(DataType::UInt32) => apply::<UInt32Type>(df, py, lambda, 0, None)?.into_series(),
                Some(DataType::UInt64) => apply::<UInt64Type>(df, py, lambda, 0, None)?.into_series(),
                Some(DataType::Float32) => apply::<Float32Type>(df, py, lambda, 0, None)?.into_series(),
                Some(DataType::Float64) => apply::<Float64Type>(df, py, lambda, 0, None)?.into_series(),
                Some(DataType::Date) => apply::<Int32Type>(df, py, lambda, 0, None)?.into_date().into_series(),
                Some(DataType::Datetime(tu, tz)) => apply::<Int64Type>(df, py, lambda, 0, None)?.into_datetime(tu, tz).into_series(),
                Some(DataType::Boolean) => apply_lambda_with_bool_out_type(df, py, lambda, 0, None)?.into_series(),
                Some(DataType::String) => apply_lambda_with_string_out_type(df, py, lambda, 0, None)?.into_series(),
                _ => return apply_lambda_unknown(df, py, lambda, inference_size),
            };

            Ok((PySeries::from(out).into_py_any(py)?, false))
        })
    }

    pub fn shrink_to_fit(&mut self, py: Python) -> PyResult<()> {
        py.enter_polars_ok(|| self.df.shrink_to_fit())
    }

    pub fn hash_rows(
        &mut self,
        py: Python<'_>,
        k0: u64,
        k1: u64,
        k2: u64,
        k3: u64,
    ) -> PyResult<PySeries> {
        // TODO: don't expose all these seeds.
        let seed = PlFixedStateQuality::default().hash_one((k0, k1, k2, k3));
        let hb = PlSeedableRandomStateQuality::seed_from_u64(seed);
        py.enter_polars_series(|| self.df.hash_rows(Some(hb)))
    }

    #[pyo3(signature = (keep_names_as, column_names))]
    pub fn transpose(
        &mut self,
        py: Python<'_>,
        keep_names_as: Option<&str>,
        column_names: &Bound<PyAny>,
    ) -> PyResult<Self> {
        let new_col_names = if let Ok(name) = column_names.extract::<Vec<String>>() {
            Some(Either::Right(name))
        } else if let Ok(name) = column_names.extract::<String>() {
            Some(Either::Left(name))
        } else {
            None
        };
        py.enter_polars_df(|| self.df.transpose(keep_names_as, new_col_names))
    }

    pub fn upsample(
        &self,
        py: Python<'_>,
        by: Vec<String>,
        index_column: &str,
        every: &str,
        stable: bool,
    ) -> PyResult<Self> {
        let every = Duration::try_parse(every).map_err(PyPolarsErr::from)?;
        py.enter_polars_df(|| {
            if stable {
                self.df.upsample_stable(by, index_column, every)
            } else {
                self.df.upsample(by, index_column, every)
            }
        })
    }

    pub fn to_struct(
        &self,
        py: Python<'_>,
        name: &str,
        invalid_indices: Vec<usize>,
    ) -> PyResult<PySeries> {
        py.enter_polars_series(|| {
            let mut ca = self.df.clone().into_struct(name.into());

            if !invalid_indices.is_empty() {
                let mut validity = MutableBitmap::with_capacity(ca.len());
                validity.extend_constant(ca.len(), true);
                for i in invalid_indices {
                    validity.set(i, false);
                }
                ca.rechunk_mut();
                Ok(ca.with_outer_validity(Some(validity.freeze())))
            } else {
                Ok(ca)
            }
        })
    }

    pub fn clear(&self, py: Python) -> PyResult<Self> {
        py.enter_polars_df(|| Ok(self.df.clear()))
    }

    /// Export the columns via polars-ffi
    /// # Safety
    /// Needs a preallocated *mut SeriesExport that has allocated space for n_columns.
    pub unsafe fn _export_columns(&mut self, location: usize) {
        use polars_ffi::version_0::export_column;

        let cols = self.df.get_columns();

        let location = location as *mut SeriesExport;

        for (i, col) in cols.iter().enumerate() {
            let e = export_column(col);
            // SAFETY:
            // Caller should ensure address is allocated.
            // Be careful not to drop `e` here as that should be dropped by the ffi consumer
            unsafe { core::ptr::write(location.add(i), e) };
        }
    }

    /// Import [`Self`] via polars-ffi
    /// # Safety
    /// [`location`] should be an address that contains [`width`] properly initialized
    /// [`SeriesExport`]s
    #[classmethod]
    pub unsafe fn _import_columns(
        _cls: &Bound<PyType>,
        location: usize,
        width: usize,
    ) -> PyResult<Self> {
        use polars_ffi::version_0::import_df;

        let location = location as *mut SeriesExport;

        let df = unsafe { import_df(location, width) }.map_err(PyPolarsErr::from)?;
        Ok(PyDataFrame { df })
    }

    /// Internal utility function to allow direct access to the row encoding from python.
    #[pyo3(signature = (opts))]
    fn _row_encode(&self, py: Python<'_>, opts: Vec<(bool, bool, bool)>) -> PyResult<PySeries> {
        py.enter_polars_series(|| {
            let name = PlSmallStr::from_static("row_enc");
            let is_unordered = opts.first().is_some_and(|(_, _, v)| *v);

            let ca = if is_unordered {
                _get_rows_encoded_ca_unordered(name, self.df.get_columns())
            } else {
                let descending = opts.iter().map(|(v, _, _)| *v).collect::<Vec<_>>();
                let nulls_last = opts.iter().map(|(_, v, _)| *v).collect::<Vec<_>>();

                _get_rows_encoded_ca(
                    name,
                    self.df.get_columns(),
                    descending.as_slice(),
                    nulls_last.as_slice(),
                )
            }?;

            Ok(ca)
        })
    }
}
