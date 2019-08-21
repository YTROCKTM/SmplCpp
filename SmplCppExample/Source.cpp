#include "BodyModel.h"

int main(int argc, char** argv)
{
	BodyModel smpl;
	smpl.load("./smpl/ModelTxt_M/0.04/");
	Eigen::VectorXf shape(10);
	shape.setConstant(0.f);
	smpl.changeShapeAB(shape);
	smpl.saveAsObj("./smpl.obj");
	return 0;
}