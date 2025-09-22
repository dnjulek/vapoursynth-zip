#include <stdio.h>
#include <math.h>

float powf_fast(float a, float b) {
	union { float d; int x; } u = { a };
	u.x = (int)(b * (u.x - 1064866805) + 1064866805);
	return u.d;
}

int main() {
	float test_cases[][2] = {
		{2.0f, 3.0f},
		{5.0f, 2.0f},
		{10.0f, 0.5f},
		{3.14f, 2.0f},
		{2.0f, 8.0f}
	};

	int num_tests = sizeof(test_cases) / sizeof(test_cases[0]);

	printf("Testing powf_fast vs standard powf:\n");
	printf("%-10s %-10s %-15s %-15s %-10s\n", "Base", "Exp", "powf_fast", "powf", "Error");
	printf("---------------------------------------------------------------\n");

	for (int i = 0; i < num_tests; i++) {
		float a = test_cases[i][0];
		float b = test_cases[i][1];
		float fast_result = powf_fast(a, b);
		float std_result = powf(a, b);
		float error = fabsf(fast_result - std_result) / std_result * 100.0f;

		printf("%-10.2f %-10.2f %-15.6f %-15.6f %-10.2f%%\n",
			   a, b, fast_result, std_result, error);
	}

	return 0;
}