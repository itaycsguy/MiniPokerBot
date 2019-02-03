     __  __ _       _   _____      _                        _____ 
    |  \/  (_)     (_) |  __ \    | |                 /\   |_   _|
    | \  / |_ _ __  _  | |__) |__ | | _____ _ __     /  \    | |  
    | |\/| | | '_ \| | |  ___/ _ \| |/ / _ \ '__|   / /\ \   | |  
    | |  | | | | | | | | |  | (_) |   <  __/ |     / ____ \ _| |_ 
    |_|  |_|_|_| |_|_| |_|   \___/|_|\_\___|_|    /_/    \_\_____|

Github repo: https://github.com/itaycsguy/MiniPokerBot.git

Tutorial:
=========
1. All files in the same directory & run the program from the directory location path
2. Developed and tested on python interpreter version: 3.6.6
3. If you get any exception check the installed libraries with 'pip' tool

Libraries:
----------
random,math,os,sys,numpy,itertools,operator,json,copy,time,pickle,from collections import Counter,matplotlib.pyplot

GUI Usage:
==========
Our program is highly customisable. Therefore, we have prepared four presets to help create the best model for learning and testing.
Start by entering command "6" and then choose preset "0" to start learning.
To improve your learning softbot please continue with preset "1" and to test the softbot please continue with preset "2","3"

After choosing each preset enter command 7 to simulate.

Subsquently, you will be prompt to choose 2 agents types. For now please choose "0" (Qagent) as the first player and
"1" (RandomAgent) for the second player(Note that you may also leave it blank and press enter to choose the default values)

Once the simulation is complete, a summary will be displayed above the menu.

Preset 0 is the first stage that creates the qtable with random states, by dealing random cards.
Preset 1 is the extended learning which is done by dealing cards from reduced qtable after the initial learning.
Preset 2 is the high volume stage which simulate games based on the states we have learned.
Preset 3 is the display stage which help you visually inspect small game rounds from the simulation to get a sense of how our QAgent "thinks" and "acts".

Developers: Itay Guy & Dean Sharaby