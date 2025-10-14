
- [x] For each PQD, you need to select one or several features from the provided eight features to separate it from the other PQDs. Enough reasoning and results should be provided to support your selection.
- [ ] Then, based on your selection, implement a classifier to test the effectiveness of
your selection. Your classifier should have only one output which indicates the possibility of the fed sample belonging to the PQD under study.
- [ ] The dataset should be split into training set and validation set with cross validation.

| Name      | Correlations                                                        |
| --------- | ------------------------------------------------------------------- |
| Swell     | All besides 0, 3                                                    |
| Transient | (1,2), (1,5), (1,6), (1,7), <br>(2,4), (2,5), (2,6), (2,7)<br>(5,7) |
| Swell_h   | (1,4), (1,2), ()                                                    |

| PQD      | sag                                    | swell                     | interruption       | transient | harmonics | fluctuation    | sag harmonics             | swell harmonics |
| -------- | -------------------------------------- | ------------------------- | ------------------ | --------- | --------- | -------------- | ------------------------- | --------------- |
| Features | time amplitude, harmonics, harmonics 2 | time amplitude, harmonics | time amplitude,min | harmonics | harmonics | time amplitude | time amplitude, harmonics | max/min         |


![[Pasted image 20251006121118.png]]

![[Pasted image 20251006121129.png]]

![[Pasted image 20251006121139.png]]

![[Pasted image 20251006121149.png]]

![[Pasted image 20251006121201.png]]

![[Pasted image 20251006121210.png]]

![[Pasted image 20251006121221.png]]

![[Pasted image 20251006121232.png]]

![[Pasted image 20251006121240.png]]

