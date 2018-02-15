// Adapted from Apple's hello.c

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif

#include "pgm.h"

#define CHECK_ERROR_CODE(r) {if (r) {printf("An OpenCL error occurred. Error code: %d. Line: %d", r, __LINE__); return r;}}
#define CHECK_ERROR(inst) {int err = inst; CHECK_ERROR_CODE(err);}

const int TIMES = 1;

void compute_cpu(
	const unsigned char* in,
	unsigned char* out,
	int rows,
	int cols,
	int filter_size
) {
	int fs_half = filter_size / 2;
	int sum = 0, ones = 0;
	int out_rows = rows - filter_size + 1;
	int out_cols = cols - filter_size + 1;
	for (int i = 0; i < out_rows; i++) {
		for (int j = 0; j < out_cols; j++) {
			for (int r = i; r <= i + filter_size; r++) {
				for (int c = j; c < j + filter_size; c++) {
					int x_dist = j - (c - fs_half);
					if (x_dist < 0) { x_dist = -x_dist; }
					int y_dist = i - (r - fs_half);
					if (y_dist < 0) { y_dist = -y_dist; }
					if (x_dist + y_dist <= fs_half) { /* Manhattan Distance */
						ones = ones + 1;
						sum = sum + in[r*cols + c];
					}
				}
			}
			out[i*(cols - filter_size + 1) + j] = sum / ones;
		}
	}
}

const char* kernel_source[] = {
	"__kernel void compute_gpu(__global const uchar* in, __global uchar* out, int rows, int cols, int filter_size) {"
	"    int i = get_global_id(0);"
	"    int j = get_global_id(1);"
	"    int fs_half = filter_size/2;"
	"    int out_rows = rows - filter_size + 1;"
	"    int out_cols = cols - filter_size + 1;"
	"    if (i < 0 || i >= out_rows || j < 0 || j >= out_cols) {"
	"        return;"
	"    }"
	"    int r = 0, c = 0;"
	"    int sum = 0, ones = 0;"
	"    for (r = i; r <= i+filter_size; r++) {"
	"        for (c = j; c < j+filter_size; c++) {"
	"            int x_dist = j-(c-fs_half);"
	"            if (x_dist < 0) { x_dist = -x_dist; }"
	"            int y_dist = i-(r-fs_half);"
	"            if (y_dist < 0) { y_dist = -y_dist; }"
	"            if (x_dist + y_dist <= fs_half) { /* Manhattan Distance */ "
	"                ones = ones + 1;"
	"                sum = sum + in[r*cols + c];"
	"            }"
	"        }"
	"    }"
	"    out[i*(cols-filter_size+1) + j] = sum/ones;"
	"}"
};

int main(int argc, char** argv) {
	if (argc != 5) {
		printf("Usage: [this_executable] filename_in filename_out filter_size exec_mode\n\n");

		printf("exec_mode = \"--cpu\" or \"--gpu\"\n");
		return 1;
	}

	char* filename = argv[1];
	unsigned char* img = NULL;
	int rows, cols;
	if (pgm_load(&img, &rows, &cols, filename)) {
		printf("Error in loading %s\n", filename);
		return 2;
	}

	int filter_size = atoi(argv[3]);
	if (filter_size <= 0 || !(filter_size % 2)) {
		printf("Error: invalid filter size\n");
		return 3;
	}
	unsigned char* img_out = NULL;
	int fs_half = filter_size / 2;
	int out_rows = (rows - filter_size + 1);
	int out_cols = (cols - filter_size + 1);
	int out_size = out_rows*out_cols;
	img_out = malloc(sizeof(unsigned char)*out_size);
	
	int gpu = 0;
	if (!strcmp(argv[4], "--cpu")) gpu = 0;
	else if (!strcmp(argv[4], "--gpu")) gpu = 1;
	else {
		printf("Invalid exec mode.\n");
		return 4;
	}

	int start_time = 0, end_time = 0;
	// OpenCL configuration starts here
	if (gpu) {
		cl_int ret;

		/* 1) Context object */
		cl_platform_id platform_id = NULL;
		cl_device_id device_id = NULL;
		cl_context context = NULL;

		/* 2) Command queue */
		cl_command_queue command_queue = NULL;

		/* 3) Memory object */
		cl_mem memobj_in = NULL;
		cl_mem memobj_out = NULL;

		/* 4) Program object */
		cl_program program = NULL;

		/* 5) Kernel object */
		cl_kernel kernel = NULL;

		/* Get available GPU device info */
		CHECK_ERROR(clGetPlatformIDs(1, &platform_id, NULL));
		CHECK_ERROR(clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_GPU, 1, &device_id, NULL));

		/* Create OpenCL context */
		context = clCreateContext(NULL, 1, &device_id, NULL, NULL, &ret);
		CHECK_ERROR_CODE(ret);

		/* Create memory buffer */
		//memobj_in = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, rows*cols * sizeof(char), (void*)img, &ret);
		//CHECK_ERROR_CODE(ret);
		memobj_out = clCreateBuffer(context, CL_MEM_WRITE_ONLY, out_size * sizeof(char), NULL, &ret);
		CHECK_ERROR_CODE(ret);

		/* Create kernel program from the source */
		program = clCreateProgramWithSource(context, 1, kernel_source, NULL, &ret);
		CHECK_ERROR_CODE(ret);

		/* Build kernel program */
		ret = clBuildProgram(program, 1, &device_id, NULL, NULL, NULL);
		CHECK_ERROR_CODE(ret);

		/* Create OpenCL kernel */
		kernel = clCreateKernel(program, "compute_gpu", &ret);
		CHECK_ERROR_CODE(ret);

		/* Set OpenCL kernel parameters */
		//ret = clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *)&memobj_in);
		ret = clSetKernelArg(kernel, 1, sizeof(cl_mem), (void *)&memobj_out);
		ret = clSetKernelArg(kernel, 2, sizeof(int), (void*)&rows);
		ret = clSetKernelArg(kernel, 3, sizeof(int), (void*)&cols);
		ret = clSetKernelArg(kernel, 4, sizeof(int), (void*)&filter_size);
		CHECK_ERROR_CODE(ret);

		/* Create Command Queue */
		command_queue = clCreateCommandQueue(context, device_id, 0, &ret);
		CHECK_ERROR_CODE(ret);

		/* Execute OpenCL kernel */
		cl_int global_dim[2] = { out_rows, out_cols };
		start_time = clock();
		for (int i = 0; i < TIMES; i++) {
			memobj_in = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, rows*cols * sizeof(char), (void*)img, &ret);
			CHECK_ERROR_CODE(ret);
			ret = clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *)&memobj_in);
			CHECK_ERROR_CODE(ret);
			ret = clEnqueueNDRangeKernel(command_queue, kernel, 2, NULL, global_dim, 0, NULL, NULL, NULL);
			CHECK_ERROR_CODE(ret);

			/* Copy results from device to host */
			ret = clEnqueueReadBuffer(command_queue, memobj_out, CL_TRUE, 0,
				out_size * sizeof(char), img_out, 0, NULL, NULL);
			CHECK_ERROR_CODE(ret);
		}
		end_time = clock();

		/* Release resources */
		ret = clReleaseKernel(kernel);
		ret = clReleaseProgram(program);
		ret = clReleaseMemObject(memobj_in);
		ret = clReleaseMemObject(memobj_out);
		ret = clReleaseCommandQueue(command_queue);
		ret = clReleaseContext(context);
	}
	else {
		start_time = clock();
		for (int i = 0; i < TIMES; i++) compute_cpu(img, img_out, rows, cols, filter_size);
		end_time = clock();
	}
	printf("Elapsed time: %d ms", end_time - start_time);

	pgm_save(img_out, rows - filter_size + 1, cols - filter_size + 1, argv[2]);

	free(img);
	free(img_out);

	return 0;
}
