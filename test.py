import polars as pl

a = pl.select(pl.lit(-1, dtype=pl.Int32).cast(pl.UInt32, strict=False), eager=False)
a.show_graph()
