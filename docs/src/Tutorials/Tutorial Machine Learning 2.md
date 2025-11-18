# Tutorial ML 2

A common task in [BCI](@ref "Acronyms") reasearch is to test a machine learning model (MLM) on a large amount of real data.
This tutorial uses the FII-BCI corpus in [NY format](@ref) as an example.

The tutorial shows how to

1. Select databases and sessions from the FII-BCI corpus according to:
    - BCI Paradigm (Motor Imagery or P300)
    - availability of specific classes
    - minimum number of trials per class
2. Run a cross-validation for all selected [sessions](@ref "session") in all selected [databases](@ref "database") and show a summary of the cross-validation results for each session.

!!! info
    As a MLM, the [MDM](https://marco-congedo.github.io/PosDefManifoldML.jl/stable/mdm/) Riemannian classifier employing the affine-invariant (Fisher-Rao) metric is used [barachant2012multi](@cite), [Congedo2017Review](@cite). As a covariance matrix estimator, the linear shrinkage estimator of [LedoitWolf2004](@cite) is used. These are state-of-the art settings used as default in **Eegle**. 

    For each session, an 8-fold stratified cross-validation is run. The summary of results comprises the mean and standard deviation of the
    balanced accuracy obtained across the folds as well as the p-value of the cross-validation test-statistic.

---

Tell julia you want to use the Eegle package

```julia
using Eegle 
```

Create a function to print the results. The function takes as arguments the serial number of the file in the database and the result of the cross-validation, which is a [CVres](https://marco-congedo.github.io/PosDefManifoldML.jl/stable/cv/#PosDefManifoldML.CVres) structure.

```julia
pr(f, res) = println("File ", f, 
                    ". mean(sd) balanced accuracy: ", round(res.avgAcc*100, digits=2),
                    "% (± ", round(res.stdAcc*100, digits=2), "%); ", 
                    "p-value: ", round(res.p; digits = 4))
```

Select all motor imagery databases in the FII-BCI corpus featuring the "feet" and "right_hand" class. 
Within these databases, select the sessions featuring at least 30 trials for each of these classes — see [selectDB](@ref).

```julia
MIDir = joinpath(homedir(), "FII-BCI Corpus","NY", "MI") # path to MI databases
classes = ["feet", "right_hand"]
DBs = selectDB(MIDir, :MI; classes, minTrials = 30);
```

Perform the cross-validation on all selected sessions for all selected databases:

```julia
for (db, DB) ∈ enumerate(DBs)
    println("\nDatabase: ", DB.dbName)
    for (f, file) ∈ enumerate(DB.files)
        pr(f, crval(file; bandPass=(8, 32), classes))
    end
end
```

---------

Perform the cross-validation on all available P300 databases and on all sessions featuring at least 25 trials for both the `target` and `non-target` classes. For P300 there is no need to specify these two classes as they are the default:

```julia
P300Dir = joinpath(homedir(), "FII corpus","NY","P300") # path to P300 databases
DBs = selectDB(P300Dir, :P300; minTrials = 25);

for (db, DB) ∈ enumerate(DBs)
    println("\nDatabase: ", DB.dbName)
    for (f, file) ∈ enumerate(DB.files)
        pr(f, crval(file; bandPass=(1, 24)))
    end
end
```

For all possible options in running cross-validations, see [`crval`](@ref).

!!! tip
    Do not use Julia's `@threaded` macro in the internal loops above as function `crval` is already multi-threaded across folds.
