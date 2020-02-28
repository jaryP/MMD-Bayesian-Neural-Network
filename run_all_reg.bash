for i in $(find ./experiments/regression -type f); do
    echo "$i"
    python main_regression.py "$i"
done
