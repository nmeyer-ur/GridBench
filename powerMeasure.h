#ifndef _POWER_MEAS_H
#define _POWER_MEAS_H

#include "pwr.h"

#define PowerAPI_INIT\
	PWR_Cntxt cntxt = NULL;\
	PWR_Obj obj = NULL;\
	int rc;\
	double ave_power = 0.0;\
    rc = PWR_CntxtInit(PWR_CNTXT_FX1000, PWR_ROLE_APP, "app", &cntxt);\
    if (rc != PWR_RET_SUCCESS) {\
        printf("CntxtInit Failed\n");\
        return 1;\
    }\
    rc = PWR_CntxtGetObjByName(cntxt, "plat.node", &obj);\
    if (rc != PWR_RET_SUCCESS) {\
        printf("CntxtGetObjByName Failed\n");\
        return 1;\
    }\

#define PowerAPI_START(region)\
	double energy1##region = 0.0;\
	double energy1_est##region = 0.0;\
	PWR_Time ts1##region = 0;\
	PWR_Time ts1_est##region = 0;\
    rc = PWR_ObjAttrGetValue(obj, PWR_ATTR_MEASURED_ENERGY, &energy1##region, &ts1##region);\
    rc = PWR_ObjAttrGetValue(obj, PWR_ATTR_ENERGY, &energy1_est##region, &ts1_est##region);\
    if (rc != PWR_RET_SUCCESS) {\
        printf("ObjAttrGetValue Failed (rc = %d)\n", rc);\
        return 1;\
    }\

#define PowerAPI_STOP(region)\
	double energy2##region = 0.0;\
	double energy2_est##region = 0.0;\
	PWR_Time ts2##region = 0;\
	PWR_Time ts2_est##region = 0;\
    rc = PWR_ObjAttrGetValue(obj, PWR_ATTR_MEASURED_ENERGY, &energy2##region, &ts2##region);\
    rc = PWR_ObjAttrGetValue(obj, PWR_ATTR_ENERGY, &energy2_est##region, &ts2_est##region);\
    if (rc != PWR_RET_SUCCESS) {\
        printf("ObjAttrGetValue Failed (rc = %d)\n", rc);\
        return 1;\
    }\

#define PowerAPI_GET_POWER(region)\
	(( energy2##region - energy1##region ) / ((ts2##region - ts1##region) / 1000000000.0))

#define PowerAPI_GET_ENERGY(region)\
	(energy2##region-energy1##region)

#define PowerAPI_GET_TIME(region)\
	((ts2##region-ts1##region)/ 1000000000.0)\

#define PowerAPI_GET_POWER_EST(region)\
	(( energy2_est##region - energy1_est##region ) / ((ts2_est##region - ts1_est##region) / 1000000000.0))

#define PowerAPI_GET_ENERGY_EST(region)\
	(energy2_est##region-energy1_est##region)

#define PowerAPI_GET_TIME_EST(region)\
	((ts2_est##region-ts1_est##region)/ 1000000000.0)\


#define PowerAPI_CLOSE\
    PWR_CntxtDestroy(cntxt);


#endif
