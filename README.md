# DataBricks_PROJECTS
To repozytorium zawiera pliki z projektów data-engineeringowych na platformie databricks.

https://dbc-bcc79a29-5dbc.cloud.databricks.com/browse/folders/3972128235345815?o=7474650486611832


## PROJEKT 1 - Przykladowy ETL Pipeline.

Projekt 1 polega na zrobieniu pipeline ETL, który przetwarza syntetycznie wygenerowane tabelki zgodnie ze standardami modelu Medallion.

### Schemat postepowania 

1. Wygeneruj T = 20 datasetów (dataset_t) w formacie .csv ze zmienną numeryczną Y oraz zmiennymi
   Age, Height, Weight, TotalMoney, Edu, Sex, AvgBpm. Kazdy dataset ma od 500 do 1000 obserwacji
2. W datasecie zachodzi zaleznosc Y = X*beta + eps, przy czym variancja epsilona zmienia sie z czasem i jest rowna t
3. Są równiez zmienne person_id i rec_date, które nie biorą udziału w relacji (tak jakby metadane)
4. Datasety zawierają okolo 3% obserwacji z brakami danych, outlierami i wartościami bez sensu w kontekscie danej zmiennej
5. Takie datasety są przetwarzane prze ETL pipeline i po kolei
- Są one zczytywane przy uzyciu pyspark
- Tworzona jest tabelka w proj1_catalog.default o nazwie BRONZE_t, ta tabelka zawiera nieprzetworzone dane (+ date przerobienia), ale już w lekkim formacie delta table (czyli tak naprawde parquet + logi, schema on read)
- Tworzona jest tabelka w proj1_catalog.default o nazwie SILVER_t, ta tabelka do BRONZE ale oczyszczony - usuniete braki danych, outliery, zmienne kategorialne zonehotencodowane, meta_dane usunięte, data rozbita na rok i miesiąc
- Tworzona jest tabelka w proj1_catalog.default o nazwie SILVER_t, ta tabelka to SILVER ale bez danych ID, przygotowana pod regresje (same zmienne numeryczne, brak zmiennych nieuczestniczących w relacji)

- Zadanie (Job) stats_job.py wykonuje na każdej ze złotych tabelek regresję liniową, estumuje wariancję, podsumowuje jak różnią się ilości rekordów w bronze, silver i gold, zmienia te informacje w jedną tabelkę, tworzy także tabelkę gold_main, z całością danych złotych ale partycjonowaną po czasie t.


Użycie w databricks

- Spark declarative pipeline z pliku pipeline.py
- Job do ustawienia jako zadanie do wykonania po pipeline



### Opis pilków 

- create_datasets.py tworzy T datasetów
- pipeline.py przetwarza
- stats_job.py robi zadanie statystyczne
- wnioski z projektu (plik txt wymieniający zastosowane rozwiązania)
