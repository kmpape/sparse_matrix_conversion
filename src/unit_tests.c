#include <assert.h>
#include <stdio.h>
#include <stdlib.h>

#include "matrix_gen_utils.h"

// For unit tests
extern c_int arr_ind(const c_int i_col, const c_int i_row, const c_int nrows, const c_int ncols, const enum MatFormat format);
extern void copy_matrix(const c_float * in_matrix, c_float * out_matrix, const c_int nrows, const c_int ncols);
extern c_int nonzero_elements(const c_float * in_matrix, const c_int nrows, const c_int ncols);
extern void change_major_format(const c_float * in_matrix, const enum MatFormat in_format, c_float * out_matrix,
		const enum MatFormat out_format, const c_int nrows, const c_int ncols);
extern void dense_matrix_mpy(const c_float * in_mat1, const c_int nrows1, const c_int ncols1, const enum MatFormat format1,
		  const c_float * in_mat2, const c_int nrows2, const c_int ncols2, const enum MatFormat format2,
		  c_float * out_mat, const enum MatFormat format3);

void test_nonzero_elements() {
	const c_int nrows = 4;
	const c_int ncols = 5;
	const c_int test_nnz = 5;
	c_float * test_matrix = (c_float *)calloc(nrows * ncols, sizeof(c_float));
	test_matrix[0] = -20.0;
	test_matrix[1] = -20.0;
	test_matrix[10] = 10.0;
	test_matrix[13] = 1.0;
	test_matrix[nrows * ncols - 1] = -0.00001;
	const c_int nnz = nonzero_elements(test_matrix, nrows, ncols);
	assert(nnz == test_nnz);
}

void test_copy_matrix() {
	c_int i_row, i_col;
	const c_int nrows = 4;
	const c_int ncols = 5;
	// Test copy for RowMajor format
	c_float * test_mat = random_dense_matrix(nrows, ncols, RowMajor);
	c_float * copied_test_mat = malloc(sizeof(c_float) * nrows * ncols);
	copy_matrix(test_mat, copied_test_mat, nrows, ncols);
	for (i_row = 0; i_row < nrows; i_row++) {
		for (i_col = 0; i_col < ncols; i_col++) {
			assert(test_mat[i_row * ncols + i_col] == copied_test_mat[i_row * ncols + i_col]);
			assert(test_mat[arr_ind(i_col, i_row, nrows, ncols, RowMajor)] == copied_test_mat[arr_ind(i_col, i_row, nrows, ncols, RowMajor)]);
		}
	}
	free(test_mat);
	free(copied_test_mat);
	// Test copy for ColMajor format
	test_mat = random_dense_matrix(nrows, ncols, ColMajor);
	copied_test_mat = malloc(sizeof(c_float) * nrows * ncols);
	copy_matrix(test_mat, copied_test_mat, nrows, ncols);
	for (i_row = 0; i_row < nrows; i_row++) {
		for (i_col = 0; i_col < ncols; i_col++) {
			assert(test_mat[i_col * nrows + i_row] == copied_test_mat[i_col * nrows + i_row]);
			assert(test_mat[arr_ind(i_col, i_row, nrows, ncols, ColMajor)] == copied_test_mat[arr_ind(i_col, i_row, nrows, ncols, ColMajor)]);
		}
	}
	free(test_mat);
	free(copied_test_mat);
}

void test_change_major_format() {
	c_int i_row, i_col;
	const c_int nrows = 4;
	const c_int ncols = 5;
	// Test copy for RowMajor format
	c_float * test_mat = random_dense_matrix(nrows, ncols, RowMajor);
	c_float * col_major_mat = malloc(sizeof(c_float) * nrows * ncols);
	c_float * row_major_mat = malloc(sizeof(c_float) * nrows * ncols);
	change_major_format(test_mat, RowMajor, col_major_mat, ColMajor, nrows, ncols);
	change_major_format(col_major_mat, ColMajor, row_major_mat, RowMajor, nrows, ncols);
	for (i_row = 0; i_row < nrows; i_row++) {
		for (i_col = 0; i_col < ncols; i_col++) {
			assert(test_mat[arr_ind(i_col, i_row, nrows, ncols, RowMajor)] == col_major_mat[arr_ind(i_col, i_row, nrows, ncols, ColMajor)]);
			assert(row_major_mat[arr_ind(i_col, i_row, nrows, ncols, RowMajor)] == col_major_mat[arr_ind(i_col, i_row, nrows, ncols, ColMajor)]);
		}
	}
	free(test_mat);
	free(col_major_mat);
	free(row_major_mat);
}

void test_dense_to_csr_matrix() {
	c_int i_row, i_col, i_trial;
	const c_int N_TRIALS = 3;
	const c_int row_cols[3][2] = {{4,5}, {5,4}, {5,5}};
	for (i_trial = 0; i_trial < N_TRIALS; i_trial++) {
		const c_int nrows = row_cols[i_trial][0];
		const c_int ncols = row_cols[i_trial][1];
		c_float * test_mat = random_dense_matrix(nrows, ncols, RowMajor);
		c_float * test_mat_from_csr;
		c_int nnz_test_mat = nonzero_elements(test_mat, nrows, ncols);
		csr * test_csr = dense_to_csr_matrix(test_mat, nrows, ncols, RowMajor);
		assert(test_csr->nzmax == nnz_test_mat);
		assert(test_csr->nz == nnz_test_mat);
		test_mat_from_csr = csr_to_dense_matrix(test_csr, ColMajor);
		for (i_row = 0; i_row < nrows; i_row++) {
			for (i_col = 0; i_col < ncols; i_col++) {
				assert(test_mat[arr_ind(i_col, i_row, nrows, ncols, RowMajor)] == test_mat_from_csr[arr_ind(i_col, i_row, nrows, ncols, ColMajor)]);
			}
		}
		free(test_mat);
		free(test_mat_from_csr);
		free(test_csr->x);
		free(test_csr->i);
		free(test_csr->p);
		free(test_csr);
	}
	for (i_trial = 0; i_trial < N_TRIALS; i_trial++) { // again using sparsified matrices
		const c_int nrows = row_cols[i_trial][0];
		const c_int ncols = row_cols[i_trial][1];
		c_float * test_mat = random_dense_matrix(nrows, ncols, RowMajor);
		sparsify_dense_matrix(test_mat, nrows, ncols);
		c_float * test_mat_from_csr;
		c_int nnz_test_mat = nonzero_elements(test_mat, nrows, ncols);
		csr * test_csr = dense_to_csr_matrix(test_mat, nrows, ncols, RowMajor);
		assert(test_csr->nzmax == nnz_test_mat);
		assert(test_csr->nz == nnz_test_mat);
		test_mat_from_csr = csr_to_dense_matrix(test_csr, ColMajor);
		for (i_row = 0; i_row < nrows; i_row++) {
			for (i_col = 0; i_col < ncols; i_col++) {
				assert(test_mat[arr_ind(i_col, i_row, nrows, ncols, RowMajor)] == test_mat_from_csr[arr_ind(i_col, i_row, nrows, ncols, ColMajor)]);
			}
		}
		free(test_mat);
		free(test_mat_from_csr);
		free(test_csr->x);
		free(test_csr->i);
		free(test_csr->p);
		free(test_csr);
	}
}

void test_dense_to_csc_matrix() {
	c_int i_row, i_col, i_trial;
	const c_int N_TRIALS = 3;
	const c_int row_cols[3][2] = {{4,5}, {5,4}, {5,5}};
	for (i_trial = 0; i_trial < N_TRIALS; i_trial++) {
		const c_int nrows = row_cols[i_trial][0];
		const c_int ncols = row_cols[i_trial][1];
		c_float * test_mat = random_dense_matrix(nrows, ncols, RowMajor);
		c_float * test_mat_from_csc;
		c_int nnz_test_mat = nonzero_elements(test_mat, nrows, ncols);
		csc * test_csc = dense_to_csc_matrix(test_mat, nrows, ncols, RowMajor);
		assert(test_csc->nzmax == nnz_test_mat);
		assert(test_csc->nz == nnz_test_mat);
		test_mat_from_csc = csc_to_dense_matrix(test_csc, ColMajor);
		for (i_row = 0; i_row < nrows; i_row++) {
			for (i_col = 0; i_col < ncols; i_col++) {
				assert(test_mat[arr_ind(i_col, i_row, nrows, ncols, RowMajor)] == test_mat_from_csc[arr_ind(i_col, i_row, nrows, ncols, ColMajor)]);
			}
		}
		free(test_mat);
		free(test_mat_from_csc);
		free(test_csc->x);
		free(test_csc->i);
		free(test_csc->p);
		free(test_csc);
	}
	for (i_trial = 0; i_trial < N_TRIALS; i_trial++) {
		const c_int nrows = row_cols[i_trial][0];
		const c_int ncols = row_cols[i_trial][1];
		c_float * test_mat = random_dense_matrix(nrows, ncols, RowMajor);
		sparsify_dense_matrix(test_mat, nrows, ncols);
		c_float * test_mat_from_csc;
		c_int nnz_test_mat = nonzero_elements(test_mat, nrows, ncols);
		csc * test_csc = dense_to_csc_matrix(test_mat, nrows, ncols, RowMajor);
		assert(test_csc->nzmax == nnz_test_mat);
		assert(test_csc->nz == nnz_test_mat);
		test_mat_from_csc = csc_to_dense_matrix(test_csc, ColMajor);
		for (i_row = 0; i_row < nrows; i_row++) {
			for (i_col = 0; i_col < ncols; i_col++) {
				assert(test_mat[arr_ind(i_col, i_row, nrows, ncols, RowMajor)] == test_mat_from_csc[arr_ind(i_col, i_row, nrows, ncols, ColMajor)]);
			}
		}
		free(test_mat);
		free(test_mat_from_csc);
		free(test_csc->x);
		free(test_csc->i);
		free(test_csc->p);
		free(test_csc);
	}
}

void test_dense_matrix_mpy() {
	int i_col, i_row;
	const int nrows1 = 3;
	const int ncols1 = 3;
	const int nrows2 = 3;
	const int ncols2 = 4;
	const c_float in_mat_1[9] = {1,2,3, 1,2,3, 1,2,3};
	const c_float in_mat_2[12] = {1,2,3,4, 1,2,3,4, 1,2,3,4};
	c_float out_mat[12];
	const c_float result[12] = {6,12,18,24, 6,12,18,24, 6,12,18,24};

	dense_matrix_mpy(in_mat_1, nrows1, ncols1, RowMajor,
					 in_mat_2, nrows2, ncols2, RowMajor,
					 out_mat, RowMajor);

	for (i_row = 0; i_row < nrows1; i_row++) {
		for (i_col = 0; i_col < ncols2; i_col++) {
			assert(out_mat[i_row * ncols2 + i_col] == result[i_row * ncols2 + i_col]);
		}
	}
}

void run_unit_tests() {
	int count = 1;
	printf("Test (%d) test_copy_matrix() +++\n", count);
	test_copy_matrix();
	printf("Test test_copy_matrix() passed.\n");
	count ++;
	printf("Test (%d) test_nonzero_elements() +++\n", count);
	test_nonzero_elements();
	printf("Test test_nonzero_elements() passed.\n");
	count ++;
	printf("Test (%d) test_change_major_format() +++\n", count);
	test_change_major_format();
	printf("Test test_change_major_format() passed.\n");
	count ++;
	printf("Test (%d) test_dense_to_csr_matrix() +++\n", count);
	test_dense_to_csr_matrix();
	printf("Test test_dense_to_csr_matrix() passed.\n");
	count ++;
	printf("Test (%d) test_dense_to_csc_matrix() +++\n", count);
	test_dense_to_csc_matrix();
	printf("Test test_dense_to_csc_matrix() passed.\n");
	count ++;
	printf("Test (%d) test_dense_matrix_mpy() +++\n", count);
	test_dense_matrix_mpy();
	printf("Test test_dense_matrix_mpy() passed.\n");
	count ++;
}
