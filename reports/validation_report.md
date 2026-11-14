# Data Validation Report

## users

- Rows in: 943
- Rows passed downstream: 943
- Rows logged to error dataset: 10

| Rule | Violations |
|---|---|
| wrong_dtype_age | 10 |

## items

- Rows in: 1682
- Rows passed downstream: 1682
- Rows logged to error dataset: 0

| Rule | Violations |
|---|---|
| none | 0 |

## interactions

- Rows in: 100060
- Rows passed downstream: 99866
- Rows logged to error dataset: 194

| Rule | Violations |
|---|---|
| duplicate_row | 60 |
| missing_value | 39 |
| invalid_rating | 30 |
| invalid_timestamp | 25 |
| orphan_user_id | 20 |
| orphan_item_id | 20 |

Note: a row can violate several rules, so rule counts may exceed row counts. Users with only an invalid age are repaired (age blanked, imputed later) and kept.