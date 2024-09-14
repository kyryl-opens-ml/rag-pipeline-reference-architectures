# rag-pipeline-reference-architectures


## CLI 

To run simple CLI RAG:

```
python rag_pipeline/simple_rag.py
```

## Dagster

To run ingestion part as Dagster:


```
dagster dev -f rag_pipeline/dagster_rag_pipeline.py -p 3000 -h 0.0.0.0
```

TODO: 
- add https://dagster.io/blog/dagster-asset-checks
- add https://github.com/explodinggradients/ragas


## AirFlow

```
export AIRFLOW_HOME=$PWD/rag_pipeline/airflow-home/
export AIRFLOW__CORE__LOAD_EXAMPLES=False
airflow standalone
```

TODO:
- add https://github.com/explodinggradients/ragas