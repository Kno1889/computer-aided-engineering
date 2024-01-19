
#include <stdio.h>
#include <math.h> // only for log calculation for error comparison

float calculatePartialSum_kahan(int lower_limit, int upper_limit);

void main()
{
    unsigned int N = 1e9; // number of terms

    int quarter = N / 4;
    int halfway = N / 2;
    int three_fourths = 3 * N / 4;

    float processor_1 = 0.0;
    float processor_2 = 0.0;
    float processor_3 = 0.0;
    float processor_4 = 0.0;

    float half_sum_1 = 0.0;
    float half_sum_2 = 0.0;

    processor_1 = calculatePartialSum_kahan(1, quarter);             // from 2 to quarter - 1
    processor_2 = calculatePartialSum_kahan(quarter, halfway);       // from quarter to halfway - 1
    processor_3 = calculatePartialSum_kahan(halfway, three_fourths); // from halfway to three_fourths - 1
    processor_4 = calculatePartialSum_kahan(three_fourths, N + 1);   // from three_fourths to n

    half_sum_1 = processor_1 + processor_2;
    half_sum_2 = processor_3 + processor_4;

    float sum = half_sum_1 + half_sum_2;

    printf("\n");
    printf("------------------------------------------------------\n");
    printf("\n");
    printf("Ln2 Kahan approximation with %d terms: %.10f\n", N, sum);
    printf("------------------------------------------------------\n");
    printf("\n");
    printf("Ln2 approximation using C built-in log function: %.10f\n", log(2));
    printf("\n");
    printf("------------------------------------------------------\n");
    printf("\n");
    printf("Error: %.10f\n", (float)fabs(sum - log(2)));
}

float calculatePartialSum_kahan(int lower_limit, int upper_limit)
{
    float partial_sum = 0.0;
    float error = 0.0;
    float temp = 0.0;
    int sign;

    for (int n = lower_limit; n < upper_limit; n++)
    {
        if (n % 2 == 0)
        {
            sign = -1;
        }
        else
        {
            sign = 1;
        }
        float term = (float)((1 / (float)n) * sign - error);

        temp = partial_sum + term;
        error = (temp - partial_sum) - term;

        partial_sum = temp;
    }

    return partial_sum;
}
