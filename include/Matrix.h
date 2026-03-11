// Adapted from MPCS 51044 course material, Week 4 lecture notes,
// University of Chicago, Winter 2026.

#ifndef MATRIX_H
#  define MATRIX_H
#include <initializer_list>
#include <algorithm>
#include <array>
#include <numeric>
#include <iostream>
#include <sstream>
#include <iomanip>
#include <concepts>
#include <stdexcept>
#include <memory>

#undef minor
using std::initializer_list;
using std::array;
using std::accumulate;
using std::ostream;
using std::endl;
using std::ostringstream;
using std::streamsize;
using std::setw;
using std::max;
using std::floating_point;

namespace matrix_engine {

template<floating_point T, int rows, int cols = rows>
class Matrix {
public:
	Matrix() : data(std::make_unique<array<array<T, cols>, rows>>()) {}
	Matrix(initializer_list<initializer_list<T>> init) : data(std::make_unique<array<array<T, cols>, rows>>()) {
		if (init.size() > static_cast<size_t>(rows)) {
			throw std::invalid_argument("Matrix initializer has too many rows");
		}

		size_t rowIndex = 0;
		for (auto const &row : init) {
			if (row.size() > static_cast<size_t>(cols)) {
				throw std::invalid_argument("Matrix initializer has too many columns");
			}
			std::copy(row.begin(), row.end(), data->operator[](rowIndex).begin());
			++rowIndex;
		}
	}

	Matrix(Matrix &&) = default;
	Matrix &operator=(Matrix &&) = default;


	T &operator()(int x, int y) {
		if (x < 0 || x >= rows || y < 0 || y >= cols) {
			throw std::out_of_range("Matrix index out of range");
		}
		return (*data)[x][y];
	}

	T operator()(int x, int y) const {
		if (x < 0 || x >= rows || y < 0 || y >= cols) {
			throw std::out_of_range("Matrix index out of range");
		}
		return (*data)[x][y];
	}

	inline friend
		ostream &
		operator<<
			(ostream &os, Matrix<T, rows, cols> const &m) {
		size_t width = m.longestElementSize() + 2;
		os << "[ " << endl;
		for (int i = 0; i < rows; i++) {
			for (int j = 0; j < cols; j++) {
				os << setw(static_cast<streamsize>(width)) << m(i, j);
			}
			os << endl;
		}
		os << "]" << endl;
		return os;
	}

	Matrix<T, rows - 1, cols - 1> minor(int r, int c) const {
		Matrix<T, rows - 1, cols - 1> result;
		for (int i = 0; i < rows; i++) {
			if (i == r) {
				continue;
			}
			for (int j = 0; j < cols; j++) {
				if (j == c) {
					continue;
				}
				result(i < r ? i : i - 1, j < c ? j : j - 1) = (*data)[i][j];
			}
		}
		return result;
	}

	Matrix<T, cols, rows> operator+=(Matrix<T, cols, rows> const &other) {
		for (int i = 0; i < rows; i++) {
			for (int j = 0; j < cols; j++) {
				(*data)[i][j] += other(i, j);
			}
		}
		return *this;
	}

	// Defer the definition until further below to avoid
	// problems with forward references
	T determinant() const;

private:
	static size_t accumulateMax(size_t acc, T d) {
		ostringstream ostr;
		ostr << d;
		return std::max(acc, ostr.str().size());
	}
	static size_t accumulateMaxRow(size_t acc, array<T, cols> row) {
		return std::max(acc, accumulate(row.begin(), row.end(), static_cast<size_t>(0), accumulateMax));
	}
	size_t longestElementSize() const {
		return accumulate(data->begin(), data->end(), 0, accumulateMaxRow);
	}

	std::unique_ptr<std::array<std::array<T, cols>, rows>> data;

public:
	Matrix(Matrix const &) = delete;
	Matrix &operator=(Matrix const &) = delete;
};


template<floating_point T, int h, int w>
T
determinantImpl(Matrix<T, h, w> const &m)
{
	T val = 0;
	for (int i = 0; i < h; i++) {
		val += (i % 2 ? -1 : 1) * m(i, 0) * m.minor(i, 0).determinant();
	}
	return val;
}

template<floating_point T>
T
determinantImpl(Matrix<T, 1, 1> const &m)
{
	return m(0, 0);
}

template<floating_point T>
T
determinantImpl(Matrix<T, 2, 2> const &m)
{
	return m(0, 0) * m(1, 1) - m(0, 1) * m(1, 0);
}

template<floating_point T, int h, int w>
T
Matrix<T, h, w>::determinant() const
{
	return determinantImpl(*this);
}


template<floating_point T, int a, int b, int c>
inline Matrix<T, a, c>
operator*(Matrix<T, a, b> const &l, Matrix<T, b, c> const &r)
{
	Matrix<T, a, c> result;
	for (int i = 0; i < a; i++) {
		for (int j = 0; j < c; j++) {
			T total = 0;
			for (int k = 0; k < b; k++)
				total += l(i, k) * r(k, j);
			result(i, j) = total;
		}
	}
	return result;
}

template<floating_point T, int a, int b>
inline Matrix<T, a, b>
operator+(Matrix<T, a, b> const &l, Matrix<T, a, b> const &r)
{
	Matrix<T, a, b> result;
	for (int i = 0; i < a; i++) {
		for (int j = 0; j < b; j++) {
			result(i, j) = l(i, j) + r(i, j);
		}
	}
	return result;
}




} // namespace matrix_engine
#endif
