
![OCDetect](https://github.com/OCDetect/OCDetect-pipeline/assets/131669368/4786aabb-5da0-457c-bb25-c5df5079940a)


# preprocessing

## needed files

- main
- csv loader (dict: list pos -> subject id)
  - Input: 
    - all subjects
    - subset of subjects
   - Output: 
      - list of subjects
      - per subject: list of dataframes (single recordings)
   - do not forget to take care of:
      - drop duplicates (to delete corrupt file endings) -> also drop the inital value that was duplicated (set parameter keep==false)
   - add test flag / int , s.t. some files are loaded only (HOW MANY? & random selection)
- misc
   - initial hand washing time calculation
   - logger that can be used from everywhere

- filter 
  - Input: single dataframe 
  - Output: 
      - None (if file has no information)
      - preprocessed dataframe
  - functions:
      - magnitude calculation
      - movement calculation on a window (std of mag in window < threshold)      
  - rules:
      - ignore file completely, if:
         - corrupt files (empty or header only)
         - recording time is smaller the person specific inital hand washing time
         - has no movement at all (delete when remaining windows are smaller than inital hw time)
         - recording date is before inital hw recording
      - ignore regions of file (column "ignore" -> true/false(None):
         - when recording includes initial hw, delete regions that were under supervision
         - that have no movements
         - label was set too early in file that was cannot be hand washing before
      - how to treat labels:
         - after applying filter rules -> filling out ignore column
         - invalid label when movement before happened only for a certain time (tbd, initially 5s) --> and before that ignore col == true [paper with duration times](https://www.jstor.org/stable/26329601 )
         - multiple labels set in short succession 
            - ~~TBD with Karina what to do~~
            - correction if very short time and different answer values? (<= min hw duration)
            - TODO: calculate how often this occurs (per person) (kristina has it in older code?), in order to check risk if wrong decision has been made.
            - 
            - how to acknowledge: define min. fixed time (e.g. 10s) between 2 hw activities
  
  - visualization:
    - plots, plots, plots (everything we have and check if it / what we need)


- relabel
    - on basis of initial hw time
    - fixed timespan

