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
       
- misc
   - initial hand washing time calculation

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
         - invalid label when movement before happened only for a certain time (tbd, initially 5s) --> and before that ignore col == true
         - multiple labels set in short succession 
            - TBD with Karina what to do
            - how to acknowledge: define min. fixed time (e.g. 10s) between 2 hw activities
  
