# export PATH=$PATH:.

# User specific aliases and functions for all shells
alias code='code-fb-insiders'
alias obash='code-fb-insiders ~/.bashrc'
alias sbash='source ~/.bashrc'

# Print and run command.
function pr(){
    echo "$*"
    $*
}

function exe {
    name=$1
    shift # now $* doesn't include $1
    pr make && pr ./$name.exe $*
}

function run {
    pr make && pr ./$1.exe < $1.in;
}

function runi {
    pr make && pr ./$1.exe < $2;
}

function prep() {
    for arg in "$@"
    do
        pr cp $arg.cpp tmp_$arg;
        pr truncate -s 0 $arg.in;
        pr cp Template0.cpp $arg.cpp;
    done
    pr make
}

function lprep() {
    for arg in "$@"
    do
        pr cp $arg.cpp tmp_$arg;
        pr truncate -s 0 $arg.in;
        pr cp LeetcodeTemplate.cpp $arg.cpp;
    done
    pr make
}

function open() {
    for arg in "$@"
    do
        pr code-fb-insiders $arg.in;
        pr code-fb-insiders $arg.cpp;
    done
}

# https://codeforces.com/blog/entry/79024?locale=en
