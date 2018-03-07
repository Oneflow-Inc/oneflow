#include "oneflow/core/operator/basic_lstm_op.h"

namespace oneflow {

const PbMessage& BasicLstmOp::GetCustomized() const {	
	return op_cof().basic_lstm_conf();
}

}				
