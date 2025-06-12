      ______                              _
     / _____)             _              | |
    ( (____  _____ ____ _| |_ _____  ____| |__
     \____ \| ___ |    (_   _) ___ |/ ___)  _ \
     _____) ) ____| | | || |_| ____( (___| | | |
    (______/|_____)_|_|_| \__)_____)\____)_| |_|
        (C)2016 Semtech

SX1280 drivers
=======================

# Introduction
---------------
This project is the firmware development for SX1280 Development Kits

# Repository work flow
----------------------
The work flow of the repository applies to the gitflow work flow. For detailed information, please take a look at: https://github.com/nvie/gitflow or http://nvie.com/posts/a-successful-git-branching-model/.
The repository contains different branches:
* master
  The master branch contains the only stable versions which passes the LoRaWAN compliance tests. Therefore if you checkout the repo without switching the branch, you're automatically using the most recent stable version of the stack.
* develop
  The develop branch provides updates to master branch. It shows the progress of development for the repository. When the develop branch is ready for a release, a release branch will be created at this point. The repository must contain only one develop branch! In general, the develop branch is not a stable version and might not pass the LoRaWAN compliance tests.
* release
  As soon as we branch from develop to release branch, no further features will be integrated for the upcoming release. This branch is basically for polishing and fixing issues that relevant for the upcoming release only. As soon as this version passes the compliance test, it'll be merged into master branch.
* feature
  A feature branch reflects the development process on a specific feature. The repository might contain several feature branches. If a feature is ready, the related feature branch will be merged into the develop branch.

A script has been designed to help include this driver with the correct version. Check *repo-init.sh* in firmware projects to see which version of SX1280 driver is in use.



# Dependencies
---------------


# Install
---------
This project contains driver for SX1280 through a link to the git project sx1280-driver. This is achieve with the help of git submodules hence additional operations are required to gather all source files.


TL;DR: `git clone --recursive git@ch02git1.semnet.dom:sx1280/sx1280-driver.git`


## Obtention of this project code
---------------------------------
The source code for this project is accessible with
```
git clone git@ch02git1.semnet.dom:sx1280/sx1280-driver.git
```


## Obtention of driver code
---------------------------
After downloading this project, the driver code at the correct version is available with:
```
git submodule init
git submodule update
```

Note that for git client version > 1.9, all the above commands can be summarized with:
```
git clone --recursive git@ch02git1.semnet.dom:sx1280/sx1280-driver.git
```


## Manual include of sx1280 driver
----------------------------------

In case the above commands do not yield the expected result, it is possible to manually include the driver code in the project by issuing
```bash
cd Demo/
git clone git@ch02git1.semnet.dom:sx1280/sx1280-driver.git
```

It is then adviced to manually checkout the driver code to the correct version which should be a tag version, for instance:
```bash
git checkout v1.0
```
If the last known driver version if v1.0.


# Usage
---------


# Changelog
------------
See changelog.md file
