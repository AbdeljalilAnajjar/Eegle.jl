# Example Data

**Eegle** deploys example EEG data files in [NY format](@ref) to be used in examples and tutorials.
This format is read by fucntion [`readNY`](@ref), which creates an [`EEG`](@ref) structure including both the data and metadata. 
The complete path to the file is is accessible anywhere as Eegle.{CONSTANT},
where {CONSTANT} is one of the heading here below. See the [tutorials](@ref "Tutorials").

## EXAMPLE_MI_1

This is the file corresponding to the first [session](@ref) of the first [subject](@ref) 
of motor-imagery Brain-Computer Interface database [Alex MI](@ref "Database Summary for MI Paradigm")

## EXAMPLE_P300_1

This is the file corresponding to the first [session](@ref) of the first [subject](@ref) 
of P300 Brain-Computer Interface database [bi2012-T](@ref "bi2012")


## EXAMPLE_Normative_1

Eyes-close resting-state continuous EEG recording acquired on a 20yo healthy woman 
with a *Mitsar-EEG-201* amplifier (Mitsar Co. Ltd., St. Petersburg, Russia), from 19 standard Ag/AgCl electrodes (10-20 international system) at 128 samples per second.
The data have been high-pass filtered at 1.5 Hz and manually de-artefacted using the Eureka3! software (Nova Tech EEG, Inc., Mesa, AZ, USA).
The file holds the first 80s of the recording.

This file is accompanied by file `EXAMPLE_Normative_1_sensors.txt` holding the sensor labels.
