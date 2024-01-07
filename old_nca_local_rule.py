    def nca_local_rule(self, param, hidden_states): 
        torch.autograd.set_detect_anomaly(True)
        updates = torch.zeros(param.shape[0], param.shape[1], 1+hidden_states.shape[2]).to(self.device)
        idx_in_units = torch.randperm(param.shape[0])[:int(param.shape[0]*self.prop_cells_updated)][:,None]
        idx_out_units = torch.randperm(param.shape[1])[:int(param.shape[1]*self.prop_cells_updated)][:,None]
        for i in idx_in_units:
            for j in idx_out_units:
                if i == 0:
                    forward = param[i+1:, j].flatten()
                    forward_states = hidden_states[i+1:,j,:]
                elif i == param.shape[0]:
                    forward = param[:i, j].flatten()
                    forward_states = hidden_states[:i,j,:]
                else:
                    forward = torch.cat((param[:i,j].flatten(), param[i+1:, j].flatten()))
                    forward_states = torch.cat((hidden_states[:i,j], hidden_states[i+1:, j]))
                if j == 0:
                    backward = param[i, j+1:].flatten()
                    backward_states = hidden_states[i,j+1:,:]
                elif j == param.shape[1]:
                    backward = param[i, :j].flatten()
                    backward_states = hidden_states[i,:j,:]
                else:
                    backward = torch.cat((param[i,:j].flatten(), param[i, j+1:].flatten()))
                    backward_states = torch.cat((hidden_states[i,:j], hidden_states[i, j+1:]), dim=1)
                concatted_weights = torch.cat((param[i,j], torch.mean(forward)[None], torch.mean(backward)[None]))
                concatted_states = torch.cat((hidden_states[i,j, :].flatten(), torch.mean(forward_states, dim=0).flatten(), torch.mean(backward_states, dim=1).flatten()))
                concatted = torch.cat((concatted_weights, concatted_states))
                updates[i,j, :] = self.local_nn(concatted)
        weight_updates = updates[:,:,0] 
        hidden_state_updates = updates[:,:,1:]
        return weight_updates, hidden_state_updates
