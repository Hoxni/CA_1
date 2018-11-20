#define _CRT_SECURE_NO_WARNINGS
#define _SCL_SECURE_NO_WARNINGS

#include <stdio.h>
#include <iostream>
#include "C:\Program Files\Microsoft HPC Pack 2012\Inc\mpi.h"
#include <sys\timeb.h>
#include <Windows.h>
#include <fstream>
#include <ctime>

double *matrix = nullptr;
double *x_old = nullptr;
double *x = nullptr;
double *b = nullptr;

int dimention = 0;
double error = 0.0000001;

int *displs = nullptr;
int *sendcounts = nullptr;

int is_end = 1;

void matrix_cast() {
	for (int i = 0; i < dimention; i++) {
		double multiple = matrix[i * dimention + i];
		for (int j = 0; j < dimention; j++) {
			matrix[i * dimention + j] /= (-1.0) * multiple;
		}
		b[i] /= multiple;
		matrix[i * dimention + i] = 0.0;
	}
}

double norm() {
	double norma = 0.0;
	for (int i = 0; i < dimention; i++) {
		norma += std::pow(x_old[i] - x[i], 2);
	}
	return std::sqrt(norma);
}

void Jacobi_Itr(int& myrank, int& size) {
	for (int i = 0; i < size; i++) {
		x[displs[myrank] + i] = b[displs[myrank] + i];
		for (int j = 0; j < dimention; j++) {
			x[displs[myrank] + i] += matrix[(displs[myrank] + i) * dimention + j] * x_old[j];
		}
	}
	return;
}

void method(int& myrank, int& ranksize, int& size) {
	x_old = new double[dimention] {0.0};
	x = new double[dimention] {0.0};

	MPI_Allgather(&size, 1, MPI_INT, sendcounts, 1, MPI_INT, MPI_COMM_WORLD);
	displs[0] = 0;
	for (int i = 1; i < ranksize; i++) {
		displs[i] = displs[i - 1] + sendcounts[i - 1];
	}

	int itr = 0;

	do {
		itr += 1;
		std::copy(x, x + dimention, x_old);

		Jacobi_Itr(myrank, size);

		MPI_Allgatherv(&x[displs[myrank]], size, MPI_DOUBLE, x, sendcounts, displs, MPI_DOUBLE, MPI_COMM_WORLD);

		if (myrank == 0)
			is_end = norm() > error;
		MPI_Bcast(&is_end, 1, MPI_INT, 0, MPI_COMM_WORLD);
	} while (is_end);

	delete[] x_old;

	if (myrank == 0)
		std::cout << "Iterations =  " << itr << std::endl;
	return;
}

int main(int argc, char ** argv)
{
	SetConsoleCP(1251);
	SetConsoleOutputCP(1251);

	int myrank, ranksize;
	MPI_Status status;


	MPI_Init(&argc, &argv);       /* initialize MPI system */
	MPI_Comm_rank(MPI_COMM_WORLD, &myrank);    /* my place in MPI system */
	MPI_Comm_size(MPI_COMM_WORLD, &ranksize);  /* size of MPI system */

	double time = MPI_Wtime();

	displs = new int[ranksize];
	sendcounts = new int[ranksize];

	if (myrank == 0)               /* I am the master */
	{
		std::ifstream in("input.txt");

		in >> dimention;
		matrix = new double[dimention * dimention]{ 0.0 };
		b = new double[dimention] {0.0};

		////////////////////////  Читаем матрицу
		for (int i = 0; i < dimention; i++) {
			for (int j = 0; j < dimention; j++)
				in >> matrix[i * dimention + j];
			in >> b[i];
		}

		////////////////////// Приводим к итерационному виду
		matrix_cast();


		////////////// Отправка размерности
		for (int i = 1; i < ranksize; i++) {
			MPI_Send(&dimention, 1, MPI_INT, i, 98, MPI_COMM_WORLD);
		}
	}
	else {
		MPI_Recv(&dimention, 1, MPI_INT, 0, 98, MPI_COMM_WORLD, &status);
		matrix = new double[dimention * dimention];
		b = new double[dimention];
	}

	MPI_Barrier(MPI_COMM_WORLD);  /* make sure all MPI tasks are running */

	///////////// Отправка матрицы и вектора B
	if (myrank == 0)               /* I am the master */
	{
		/* distribute parameter */
		for (int i = 1; i < ranksize; i++) {
			MPI_Send(b, dimention, MPI_DOUBLE, i, 18, MPI_COMM_WORLD);
			MPI_Send(matrix, dimention * dimention, MPI_DOUBLE, i, 98, MPI_COMM_WORLD);
		}
	}
	else {
		MPI_Recv(b, dimention, MPI_DOUBLE, 0, 18, MPI_COMM_WORLD, &status);
		MPI_Recv(matrix, dimention * dimention, MPI_DOUBLE, 0, 98, MPI_COMM_WORLD, &status);
	}
	MPI_Barrier(MPI_COMM_WORLD);

	////// далее
	int size = dimention / ranksize + ((dimention % ranksize) > myrank ? 1 : 0);

	method(myrank, ranksize, size);

	if (myrank == 0) {
		std::ofstream out("result.txt");
		for (int i = 0; i < dimention; i++) {
			out << "X" << i + 1 << " = " << x[i] << std::endl;
		}
	}

	delete[] matrix;
	delete[] b;
	delete[] sendcounts;
	delete[] displs;
	delete[] x;

	std::cout << "Process " << myrank << " time === " << MPI_Wtime() - time << std::endl;

	MPI_Finalize();
	return 0;
}

double realtime()			     /* returns time in seconds */
{
	struct _timeb tp;
	_ftime(&tp);
	return((double)(tp.time) * 1000 + (double)(tp.millitm));
}