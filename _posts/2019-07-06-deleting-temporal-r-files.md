---
layout: single
title:  "Deleting temporal R files"
date:   2019-07-06 12:01:24 +0200
categories: R
classes: normal
# author_profile: true
---

During the execution of a script, R creates some temporal files containing plots or session metadata that are deleted when RStudio is closed.

But, what if we are executing the scripts without RStudio?

When launching R scripts in a non-interactive mode (e.g. executing through Rscript.exe and a batch file), I have noted that R does not always delete the temp directory at the end of the script as it is supposed to do.

Maybe it just happens when the script gets an exception and is interrupted suddenly. I have not fully identified yet the cases when this behaviour happens.

If you are using R in production, or simply have some periodic tasks executed periodically, it may cause your hard drive to run out of space.
Additionally, I faced in the past some problems with scripts getting hung and never getting closed.

This is why I have got used to add this little chunk at the bottom of the scripts I am running in a non-interactive mode:

```r
if(!interactive()){
  RemoveTempFiles()
  tools::pskill(Sys.getpid())                     # Kills R process
}
```

Being the function `RemoveTempFiles()`:

```r
RemoveTempFiles <- function(){

  flog.info("REMOVING TEMP FILES...")
  library(futile.logger)

  tryCatch({
    tmpdir <- tempdir()

    # Removes files and subdirectories
    unlink(dir(tmpdir, full.names=TRUE), recursive=TRUE, force = T)    

    # Removes the directory itself
    unlink(tmpdir, recursive=TRUE, force = T)                           

    finalStatus <- ifelse(length(dir(tmpdir, full.names=TRUE))==0,
                          "TEMP FILES DELETED SUCCESSFULLY",
                          "SOMETHING WENT WRONG!")
    flog.info(finalStatus)
  },
  error=function(e){
    errorMessage <- sprintf("Error removing temp files: %s", e$message)
    flog.error(errorMessage)
  })
}
```


I am using logging functions that belong to `futile.logger` package, as I find really useful to log all the events and errors that may arise during the execution.
The `tryCatch()` function is there just in case it fails deleting the temp files, as I want to ensure the script is killed later with `pskill`.
