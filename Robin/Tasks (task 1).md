
- [x] For each PQD, you need to select one or several features from the provided eight features to separate it from the other PQDs. Enough reasoning and results should be provided to support your selection.
- [x] Then, based on your selection, implement a classifier to test the effectiveness of
your selection. Your classifier should have only one output which indicates the possibility of the fed sample belonging to the PQD under study.
- [x] The dataset should be split into training set and validation set with cross validation.

```image-layout-a

![[Pasted image 20251017124755.png]]
![[Pasted image 20251017125327.png]]
```

```image-layout-a
![[Pasted image 20251017124811.png]]
![[Pasted image 20251017125401.png]]
```

| Name      | features                   |
| --------- | -------------------------- |
| Swell     | sd, max, avg abs           |
| Transient | sd, max, sd fma, no. peaks |
| Swell_h   | sd, max, no pts 0, avg abs |
| normal    | min, max, std              |
| sag_h     | no pts 0, max, sd, avg abs |
| sag       | no pts 0, max, sd, avg abs |
| interrupt | no pts 0, max, sd avg abs, |
| harmonics | no pts 0, min, max, sd     |
| flicker   | min, max, sd, pts 0        |

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

