#!/bin/bash -e

SCRIPT_DIR=$(realpath $(dirname ${BASH_SOURCE[0]}))
cd $(dirname ${SCRIPT_DIR})
echo -e "Working directory: $(pwd)"

: ${SLEEP_TIME:=0}

error-message()
{
    echo -e "\nError! ${@}\n"
    exit -1
}

verbose-run()
{
    echo -e "\n##### Executing \"${@}\"... #####\n"
    sleep ${SLEEP_TIME}
    eval $@
}

toupper()
{
    echo "$@" | awk -F '\\|~\\|' '{print toupper($1)}';
}

get-bool()
{
    echo "${1}" | egrep -i '^(y|on|yes|true|[1-9])$' &> /dev/null && echo 1 || echo 0
}

if [ -d "$(realpath /tmp)" ]; then
    : ${TMPDIR:=/tmp}
    export TMPDIR
fi

: ${CONFIG_DIR:=$(mktemp -t -d omnitrace-test-install-XXXX)}
: ${SOURCE_DIR:=$(dirname ${SCRIPT_DIR})}
: ${ENABLE_OMNITRACE_INSTRUMENT:=1}
: ${ENABLE_OMNITRACE_AVAIL:=1}
: ${ENABLE_OMNITRACE_SAMPLE:=1}
: ${ENABLE_OMNITRACE_PYTHON:=0}
: ${ENABLE_OMNITRACE_REWRITE:=1}
: ${ENABLE_OMNITRACE_RUNTIME:=1}

usage()
{
    print_option() { printf "    --%-10s %-30s     %s (default: %s)\n" "${1}" "${2}" "${3}" "${4}"; }
    echo "Options:"
    print_option source-dir "<PATH>" "Location of source directory" "${SOURCE_DIR}"
    print_option test-rocprof-sys-instrument "0|1" "Enable testing rocprof-sys-instrument" "${ENABLE_OMNITRACE_INSTRUMENT}"
    print_option test-rocprof-sys-avail "0|1" "Enable testing rocprof-sys-avail" "${ENABLE_OMNITRACE_AVAIL}"
    print_option test-rocprof-sys-sample "0|1" "Enable testing rocprof-sys-sample" "${ENABLE_OMNITRACE_SAMPLE}"
    print_option test-rocprof-sys-python "0|1" "Enable testing rocprof-sys-python" "${ENABLE_OMNITRACE_PYTHON}"
    print_option test-rocprof-sys-rewrite "0|1" "Enable testing rocprof-sys-instrument binary rewrite" "${ENABLE_OMNITRACE_REWRITE}"
    print_option test-rocprof-sys-runtime "0|1" "Enable testing rocprof-sys-instrument runtime instrumentation" "${ENABLE_OMNITRACE_RUNTIME}"
}

cat << EOF > ${CONFIG_DIR}/omnitrace.cfg
OMNITRACE_VERBOSE               = 2
OMNITRACE_PROFILE               = ON
OMNITRACE_TRACE                 = ON
OMNITRACE_USE_SAMPLING          = ON
OMNITRACE_USE_PROCESS_SAMPLING  = ON
OMNITRACE_OUTPUT_PATH           = %env{CONFIG_DIR}%/omnitrace-tests-output
OMNITRACE_OUTPUT_PREFIX         = %tag%/
OMNITRACE_SAMPLING_FREQ         = 100
OMNITRACE_SAMPLING_DELAY        = 0.05
OMNITRACE_COUT_OUTPUT           = ON
OMNITRACE_TIME_OUTPUT           = OFF
OMNITRACE_USE_PID               = OFF
EOF

export CONFIG_DIR
export OMNITRACE_CONFIG_FILE=${CONFIG_DIR}/omnitrace.cfg
verbose-run cat ${OMNITRACE_CONFIG_FILE}

while [[ $# -gt 0 ]]
do
    ARG=${1}
    shift

    VAL="$(echo ${ARG} | sed 's/=/ /1' | awk '{print $2}')"
    if [ -z "${VAL}" ]; then
        while [[ $# -gt 0 ]]
        do
            VAL=$(get-bool ${1})
            shift
            break
        done
    else
        VAL=$(get-bool ${VAL})
        ARG="$(echo ${ARG} | sed 's/=/ /1' | awk '{print $1}')"
    fi

    if [ -z "${VAL}" ]; then
        echo "Error! Missing value for argument \"${ARG}\""
        usage
        exit -1
    fi

    case "${ARG}" in
        --test-rocprof-sys-instrument)
            ENABLE_OMNITRACE_INSTRUMENT=${VAL}
            continue
            ;;
        --test-rocprof-sys-avail)
            ENABLE_OMNITRACE_AVAIL=${VAL}
            continue
            ;;
        --test-rocprof-sys-sample)
            ENABLE_OMNITRACE_SAMPLE=${VAL}
            continue
            ;;
        --test-rocprof-sys-python)
            ENABLE_OMNITRACE_PYTHON=${VAL}
            continue
            ;;
        --test-rocprof-sys-rewrite)
            ENABLE_OMNITRACE_REWRITE=${VAL}
            continue
            ;;
        --test-rocprof-sys-runtime)
            ENABLE_OMNITRACE_RUNTIME=${VAL}
            continue
            ;;
        --source-dir)
            SOURCE_DIR=${VAL}
            continue
            ;;
        *)
            echo -e "Error! Unknown option : ${ARG}"
            usage
            exit -1
            ;;
    esac
done

test-rocprof-sys-instrument()
{
    verbose-run which rocprof-sys-instrument
    verbose-run ldd $(which rocprof-sys-instrument)
    verbose-run rocprof-sys-instrument --help
}

test-rocprof-sys-avail()
{
    verbose-run which rocprof-sys-avail
    verbose-run ldd $(which rocprof-sys-avail)
    verbose-run rocprof-sys-avail --help
    verbose-run rocprof-sys-avail -a
}

test-rocprof-sys-sample()
{
    verbose-run which rocprof-sys-sample
    verbose-run ldd $(which rocprof-sys-sample)
    verbose-run rocprof-sys-sample --help
    verbose-run rocprof-sys-sample --cputime 100 --realtime 50 --hsa-interrupt 0 -TPH -- python3 ${SOURCE_DIR}/examples/python/external.py -n 5 -v 20
}

test-rocprof-sys-python()
{
    verbose-run which rocprof-sys-python
    verbose-run rocprof-sys-python --help
    verbose-run rocprof-sys-python -b -- ${SOURCE_DIR}/examples/python/builtin.py -n 5 -v 5
    verbose-run rocprof-sys-python -b -- ${SOURCE_DIR}/examples/python/noprofile.py -n 5 -v 5
    verbose-run rocprof-sys-python -- ${SOURCE_DIR}/examples/python/external.py -n 5 -v 5
    verbose-run python3 ${SOURCE_DIR}/examples/python/source.py -n 5 -v 5
}

test-rocprof-sys-rewrite()
{
    if [ -f /usr/bin/coreutils ]; then
        local LS_NAME=coreutils
        local LS_ARGS="--coreutils-prog=ls"
    else
        local LS_NAME=ls
        local LS_ARGS=""
    fi
    verbose-run rocprof-sys-instrument -e -v 1 -o ${CONFIG_DIR}/ls.inst --simulate -- ${LS_NAME}
    for i in $(find ${CONFIG_DIR}/omnitrace-tests-output/ls.inst -type f); do verbose-run ls ${i}; done
    verbose-run rocprof-sys-instrument -e -v 1 -o ${CONFIG_DIR}/ls.inst -- ${LS_NAME}
    verbose-run rocprof-sys-run -- ${CONFIG_DIR}/ls.inst ${LS_ARGS}
}

test-rocprof-sys-runtime()
{
    if [ -f /usr/bin/coreutils ]; then
        local LS_NAME=coreutils
        local LS_ARGS="--coreutils-prog=ls"
    else
        local LS_NAME=ls
        local LS_ARGS=""
    fi
    verbose-run rocprof-sys-instrument -e -v 1 --simulate -- ${LS_NAME} ${LS_ARGS}
    for i in $(find ${CONFIG_DIR}/omnitrace-tests-output/$(basename ${LS_NAME}) -type f); do verbose-run ls ${i}; done
    verbose-run rocprof-sys-instrument -e -v 1 -- ${LS_NAME} ${LS_ARGS}
}

if [ "${ENABLE_OMNITRACE_INSTRUMENT}" -ne 0 ]; then verbose-run test-rocprof-sys-instrument; fi
if [ "${ENABLE_OMNITRACE_AVAIL}" -ne 0 ]; then verbose-run test-rocprof-sys-avail; fi
if [ "${ENABLE_OMNITRACE_SAMPLE}" -ne 0 ]; then verbose-run test-rocprof-sys-sample; fi
if [ "${ENABLE_OMNITRACE_PYTHON}" -ne 0 ]; then verbose-run test-rocprof-sys-python; fi
if [ "${ENABLE_OMNITRACE_REWRITE}" -ne 0 ]; then verbose-run test-rocprof-sys-rewrite; fi
if [ "${ENABLE_OMNITRACE_RUNTIME}" -ne 0 ]; then verbose-run test-rocprof-sys-runtime; fi
