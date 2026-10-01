# 데이터

`semiconductor_etch_timeseries_3years.csv` (장비 5대 × 3년, 30분 간격, 263,040행, 약 34MB)

압축본에 CSV가 없다면 아래 명령으로 원본과 바이트 단위로 동일한 파일을 다시 만들 수 있습니다.

```bash
python scripts/generate_semiconductor_etch_data.py --years 3 --interval-minutes 30 \
  --start-date 2023-01-01 --target-defect-rate 0.2 --seed 42 \
  --output data/semiconductor_etch_timeseries_3years.csv
# SHA-256: f35cb01ebc67f380e8461b7a81218c56fba193fa49d261ad72d682c705b818c3
```
