     __  __ _       _   _____      _                        _____ 
    |  \/  (_)     (_) |  __ \    | |                 /\   |_   _|
    | \  / |_ _ __  _  | |__) |__ | | _____ _ __     /  \    | |  
    | |\/| | | '_ \| | |  ___/ _ \| |/ / _ \ '__|   / /\ \   | |  
    | |  | | | | | | | | |  | (_) |   <  __/ |     / ____ \ _| |_ 
    |_|  |_|_|_| |_|_| |_|   \___/|_|\_\___|_|    /_/    \_\_____|

Tutorial:
=========
1. Do not separate files
2. Run the application from the entry folder where the files are being
3. Developed and tested on python interpreter version: 3.6.6
4. If you get any exception check the installed libraries with 'pip' tool

Libraries:
----------
random
math
os
sys
numpy
itertools
operator
json
copy
time
pickle
from collections import Counter
matplotlib.pyplot

GUI Usage:
==========
Our program is highly customisable. Therefore, we have prepared four presets to help create the best model for testing and testing.
To choose a preset please enter command "6" and start by choosing preset "0".
Following with preset "1","2","3" respectively.

After choosing each preset enter command 7 to start the games.

Subsquently, you will be prompt to choose 2 agents types. For now please choose "0" (Qagent) as the first player and "1" (RandomAgent) for the second players(Note that you may also leave it blank and press enter to choose the default values)

Once the games are complete, a summary of the result, including total wins and other requests stats, will be printed above the menu.

Preset 0 is the first stage that create the qtable with random states, by dealing the players random cards.
Preset 1 is the extended learning which is done by dealing cards which were previously dealt in stage 0 assign better values to thier values.
Preset 2 is the high volume stage which gives an estimation of how the Qagent has preformed under "real" games constraints (no exploration)
Preset 3 is the display stage which help you visually inspect smally randomly chosen states and get a sense of the agent "thinking" and actions.

Developers: Itay Guy & Dean Sharaby