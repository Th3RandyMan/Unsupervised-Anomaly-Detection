$numberOfRuns = 50

for ($i = 1; $i -le $numberOfRuns; $i++) {
    Write-Host "Running iteration $i"
    jupyter nbconvert --execute --to notebook --inplace dirty_ppgData_maker_improved.ipynb
}