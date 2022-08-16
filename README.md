## This program was written during the Master's thesis at St. Pölten University of Applied Sciences, Summer Term 2022 
### Master thesis title: An autocorrelation analysis of distance measurements to detect differences in visual behavior between educational systems and between emmetropic and myopic children

* Note: The code does not contain any data for testing
* The master's thesis will be linked as soon as the thesis is published, since the thesis contains most of the documentation.
---
* Under the folder `diagrams` flowcharts are given, which represent the program flow.
* The program can be configured via the yaml files in the `.config` folder.
  * Students can be excluded or included from any execution via config
  * Via `app.yaml` it can be configured which student data sets should be executed. 
  * After the configuration has changed, please execute the tests which check if the configuration files are still valid.
* The program's entry point is `main.py`
* The program is configured that two datasets exists (2018, 2019)

The program was written specifically for the analyses of the data collected during the study "Wearable Technology-Driven Study of Short-sightedness in Chinese Children.". 

This program is intended to provide a basis for further analysis of the data. 

There is still a lot of room for improvements.

---
# Author

* **Josef Leonhartsberger** ([github](https://github.com/leonharj-fh/mdh_thesis_analyses_eye_tracker) | [email](mailto:dh201816@fhstp.ac.at))

---

### Tested with python versions:
* [3.9, 3.10]