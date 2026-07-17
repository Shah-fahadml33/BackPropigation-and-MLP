import numpy as np
class GradientsChecker:
    @staticmethod
    def check(model,x,y,eps=1e-7):
        orignal_theta=model.get_perams_vector()
        
        _=model.forward(x)
        ana_grades=model.backward(y)
        d_theta_ana=model.set_grads_vecter()

        num_perams=orignal_theta.shape[0]
        d_theta_num=np.zeros_like(orignal_theta)

        def evaluate_loss(theta_vec):
            model.set_perams_from_vector(theta_vec)
            pred=model.forward(x)
            return model.compute_loss(pred,y)

        for i in range(num_perams):
            t_plus=np.copy(orignal_theta)
            t_plus[i]+=eps
            loss_plus=evaluate_loss(t_plus)

            t_minus=np.copy(orignal_theta)
            t_minus[i]-=eps
            loss_minus=evaluate_loss(t_minus)

            d_theta_num[i]=(loss_plus-loss_minus)/(2*eps)
            
        model.set_perams_from_vector(orignal_theta)
        nominator=np.linalg.norm(d_theta_num-d_theta_ana)
        denom=np.linalg.norm(d_theta_num)+np.linalg.norm(d_theta_ana)

        if denom==0:
            return 0.0
        diff=nominator/denom

        return diff
        
def check_grads(model,x,y):
    print("======LAUNCHING GRADEINTS CHECCKING========")
    diff=GradientsChecker.check(model,x,y)
    print(f"Relative Difference is :{diff}")
    if diff<1e-7:
        print("succes :Every thing is correct")
    elif diff<1e-4:
        print("Wrong :Minor precision")
    else:
        print("Fianal Error : Check your implementation line by line")            