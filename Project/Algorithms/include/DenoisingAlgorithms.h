#pragma once

///	A strogly typed enum class representing the types of the denoising algorithms
enum class Denoising_Algorithms
{
	GAUSSIAN = 1,
	MEDIAN = 2, 
	AVERAGE = 3, 
	BILATERAL = 4, 
	NONE = 0 
};