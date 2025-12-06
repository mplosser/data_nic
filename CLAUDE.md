This repository will acquire data from the national information center, import it, and save it as parquest files for use in various projects.

The data is vailable here https://www.ffiec.gov/npw/FinancialReport/DataDownload as zipped csv files.
There are several datasets, we want separate parquest files for each of them.

Ideally we would download the data and incorporate information about the variables into the datasets much like stata has a variable description field.

There is a pdf here (https://www.ffiec.gov/npw/StaticData/DataDownload/NPW%20Data%20Dictionary.pdf) with information about the variables.

Similar to other repositories in my github that acquire data. It would be nice to have a download file that downlaods the raw data to data/raw a parse file that unzips the data and parses it into parquest files in data/processed (if we can apply variable descriptives this would also do that for the parquest files) and then a summarize file that summarizes the shape of the data for the files we have created (the time period covered, and the number of firms/branches/transformations over time) and finally a cleanup file taht deletes files in an effort to save space. There would then be a requirements and source readme that describes these key steps in the pipeline. 

If possible review the other data repos we have created together (data_sod data_fry9c data_call) for simialr exercises.

