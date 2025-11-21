# Tutorial ML 1

The goal of the tutorial is to show how to use the Augmented Covariance Matrices method proposed in [Carrara2024ACM](@cite). This method will be evaluated against a state-of-the-art benchmark by means of cross-validations (8-fold by default) on the [`EXAMPLE_MI_1`](@ref) Motor Imagery (MI) [BCI](@ref "Acronyms") EEG file provided with **Eegle**. 

From this file, we will use trials pertaining to classes "feet" and "right_hand".

!!! info
    The benchmark method is the standard minimum distance to mean (MDM) Riemannian classifier adopting the affine-invariant metric (default in **Eegle**). The classifier will be applied on sample covariance matrices (covtype = SCM) after the following pre-processing:
    1. resampling the data from 256 to 128 samples per second (rate = 1//2)
    2. filtering the EEG data in the band-pass region 8-32 Hz (bandPass = (8, 32))
    3. rejecting trials featuring abnormal amplitude (upperLimit = 1). 
 
```julia
using Eegle

args = (bandPass = (8, 32), upperLimit = 1, rate = 1//2, classes=["feet", "right_hand"]);
cv = crval(EXAMPLE_MI_1; covtype = SCM, args...) # or Eegle.crval(EXAMPLE_MI_1)
```

*Output you will see:*
```REPL
Performing 8-fold cross-validation...
⚂ ⚅ ⚂ ⚄ ⚀ ⚂ ⚃ ⚁
Done in 91 milliseconds

◕ Cross-Validation Accuracy
⭒  ⭒    ⭒       ⭒         ⭒
.cvType   : 8-fold
.scoring  : balanced accuracy
.modelType: MDM
.nTrials  : 35
.matSizes : 16 for all folds
.predLabels a vector of #classes vectors of predicted labels per fold
.losses     a vector of binary loss per fold
.cnfs       a confusion matrix per fold (frequencies)
.avgCnf     average confusion matrix (proportions)
.accs       a vector of accuracies, one per fold
.avgAcc   : 0.792
.z        : -4.3474
.p        : < 0.0001
.ms       : 91
```

The average balanced accuracy across fold is given in the .avgAcc field (0.74).

!!! warning
    ACM are obtained stacking to the EEG trials lagged versions of them before computing the sample covariance matrices. ACMs computed this way have each side of size *n × l*, where *n* is the number of electrodes and *l* is the number of lags, thus may become very large and, typically, are no longer positive-definite. Therefore, we will apply a Tikhonov regularization and a dimensionality-reduction retaining 99.9% of the explained variance (eVar=0.999). This is achieved passing a `Pipeline` object — see [here](https://marco-congedo.github.io/PosDefManifoldML.jl/stable/conditioners/). Everything else in the ACM method is the same as per the benchmark classifier.

!!! note
    ACMs encode not only the spatial distribution of brain dipolar sources, but also the *cross-covariance* between them for the embedded lags. Since after resampling the sampling rate of the data is 128, each EEG sample is spaced apart 1/128 = 7.8125 ms. We wll embed 10 lags (lags=10), thus we will be able to encode cross-covariance information up to 7.8125 * 10 = 78.125 ms. 

```julia
pipeline = @→ Tikhonov(1e-4) Recenter(Fisher; eVar=0.999, verbose=false) 
cvl = crval(EXAMPLE_MI_1; pipeline, lags=10, covtype = SCM, args...) 
```

*Output you will see:*
```REPL
Performing 8-fold cross-validation...
⚂ ⚄ ⚁ ⚅ ⚄ ⚀ ⚅ ⚁ 
Done in 7150 milliseconds

◕ Cross-Validation Accuracy
⭒  ⭒    ⭒       ⭒         ⭒
.cvType   : 8-fold
.scoring  : balanced accuracy
.modelType: MDM
.nTrials  : 35
.matSizes : [105, 105, 105, 104, 105, 105, 105, 105]
.predLabels a vector of #classes vectors of predicted labels per fold
.losses     a vector of binary loss per fold
.cnfs       a confusion matrix per fold (frequencies)
.avgCnf     average confusion matrix (proportions)
.accs       a vector of accuracies, one per fold
.avgAcc   : 0.885
.z        : -7.2495
.p        : < 0.0001
.ms       : 7150
```

The average accuracy across folds has raised from 0.792 to 0.885 with the settings we have used.

!!! tip
    The effective dimension of covariance matrices in each fold after lag embedding and dimensionality reduction by recentering is given in field `.matSizes`. Notice that the sizes are not necessarily equal to *n × l*, nor they must be equal across folds, because of the dimensionality reduction operated by the [Recenter](https://marco-congedo.github.io/PosDefManifoldML.jl/stable/conditioners/#PosDefManifoldML.Recenter) conditioner.

---

Now, let us appreciate what the function [`crval`](@ref) does for us by performing the cross-validation in a step-by-step manner:

**a** - load the data:
```julia
o = readNY(EXAMPLE_MI_1; args...)
```

**b** - encode the EEG trials, which are stored in o.trials, as covariance matrices:
```julia
C = covmat(o.trials; covtype = SCM)
Cl = covmat(o.trials; covtype = SCM, lags = 10)
```

**c** - call the `crval` function of package [PosDefManifoldML](https://github.com/Marco-Congedo/PosDefManifoldML.jl):
```julia
cv2 = crval(MDM(), C, o.y)
cvl2 = crval(MDM(), Cl, o.y; pipeline) # in julia, `pipeline` is the same as `pipeline=pipeline`
```

The results are identical. Let us verify:
```julia
cv.avgAcc == cv2.avgAcc # must be true
cvl.avgAcc == cvl2.avgAcc # must be true
```