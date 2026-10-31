# Routing accuracy (路由准确率)

41 test messages. First tool chosen correctly: **97.6%**.
Settings: model deepseek-flash

| Expected tool | Correct | Cases | Accuracy |
| --- | --- | --- | --- |
| save_job | 5 | 6 | 83% |
| analyze_job | 4 | 4 | 100% |
| fetch_job | 3 | 3 | 100% |
| find_jobs | 5 | 5 | 100% |
| research_company | 3 | 3 | 100% |
| build_cv | 4 | 4 | 100% |
| tailor_cv | 2 | 2 | 100% |
| plan_gaps | 3 | 3 | 100% |
| match_job | 2 | 2 | 100% |
| search_profile | 3 | 3 | 100% |
| none | 6 | 6 | 100% |

## Wrong routes

| Message | Expected | Got |
| --- | --- | --- |
| a1b2c3d4 rejected me today | save_job | none |
