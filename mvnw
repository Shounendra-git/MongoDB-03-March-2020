from pyspark.storagelevel import StorageLevel

# ── 1.  Load the incremental slice  ─────────────────────────────
df_input = spark.sql("""
    SELECT id            AS input_id ,
           input_partition_buckets,
           entity_id     AS input_entity_id,
           name          AS input_name ,
           address       AS input_address ,
           city          AS input_city ,
           state         AS input_state ,
           country       AS input_country ,
           postal_code   AS input_postal_code
    FROM   dpm_curated_epdh.tmps_entities_resolution
    WHERE  input_partition_buckets = {bucket}
""")

# ── 2.  First heavy cleansing + token key  ─────────────────────
df_input = (df_input
            .withColumn("name_cleaned",
                        trim(regexp_replace("input_name", r"[^A-Za-z0-9 ]", "")))
            .withColumn("key", split(lower(col("name_cleaned")), r"\s+"))
            .drop("name_cleaned1", "name_cleaned"))

# >>> THIS DataFrame is reused *twice* later   → cache it
df_input.persist(StorageLevel.MEMORY_AND_DISK)
# or simply: df_input.cache()

#  materialise once so it really goes to RAM
df_input.count()

# ── 3.  Generate word-ID map (small)  ──────────────────────────
df_er_input_word = spark.sql("""
    SELECT input_id, input_partition_buckets,
           explode(key) AS input_word
    FROM   df_input
""")

#  remove stop-words / suffix tags, … then de-dupe
df_er_input_word_id = (df_er_input_word
                       .filter(~col("input_word").isin(BSNS_SUFFIX_TAG_LIST))
                       .filter(~col("input_word").isin(ENGLISH_STOP_WORDS_LIST))
                       .dropDuplicates())

# ── 4.  Join with doc-level match table  ───────────────────────
df_match_docs = spark.table("match_doc")      # broadcast side

df_final = (df_input                          # left side already cached
            .join(df_match_docs, "input_doc_id", "inner")
            .withColumnRenamed("match_doc_id", "fill_match_doc_id"))

# >>> df_final is consumed by *both* UDF scoring & Hive write  → cache it
df_final.persist(StorageLevel.MEMORY_AND_DISK)
df_final.count()

# ── 5.  UDF feature engineering + ML score  ────────────────────
input_cols  = [c for c in df_final.columns if c.startswith("input_")]
match_cols  = [c for c in df_final.columns if c.startswith("match_")]

process_input_row_udf = udf(process_input_row)   # existing function
df_scored = (df_final
             .withColumn("input_col_key",  F.array(*[col(c) for c in input_cols]))
             .withColumn("match_col_key",  F.array(*[col(c) for c in match_cols]))
             .withColumn("score", process_input_row_udf("input_col_key",
                                                        "match_col_key"))
             .select("input_entity_id", "match_entity_id", "score",
                     "input_partition_buckets"))

# ── 6.  Write to Hive  ─────────────────────────────────────────
(df_scored
 .repartition("input_entity_id")
 .write.mode("append")
 .insertInto("dpm_curated_epdh.tmps_entity_resolution_score"))

# ── 7.  Clean-up  ─────────────────────────────────────────────
df_final.unpersist()
df_input.unpersist()
