cd  /PATH_TO_BRIDGEVLA/finetune/
# export COPPELIASIM_ROOT=$(pwd)/CoppeliaSim_Edu_V4_1_0_Ubuntu20_04 
# export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$COPPELIASIM_ROOT
# export QT_QPA_PLATFORM_PLUGIN_PATH=$COPPELIASIM_ROOT
export DISPLAY=:1.0
cd ./GemBench

# The command xvfb-run -a is used to run programs that require a graphical display (X server) in a headless environment, 
# such as servers or CI pipelines that don't have a physical display.
# Some programs (like GUI applications or tools using OpenGL, matplotlib, etc.) require an X display to function—even if you're not showing 
# anything on screen. xvfb-run tricks them into thinking there is a display by running them in a virtual one.
xvfb-run -a python3 server.py --port 13003 --model_epoch $1  --base_path $2
