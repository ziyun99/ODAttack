                #inverse_rescale
                grad_resize1 = input_grad[0,:,start_x:end_x,start_y:end_y]
                grad_resize1 = cv2.resize(grad_resize1.cpu().numpy().transpose(1,2,0),(width,height),cv2.INTER_CUBIC)
                if(perspective == 1):
                   perspective = 0
                   grad_resize1 = inverse_perspective_transform(grad_resize1,org,dst)
                grad_resize1 = torch.from_numpy(grad_resize1.transpose(2,0,1)).to(device)
                # grad_resize1=grad_resize
                # loss1=loss
                #####################
                #####################
                if j==0:
                   grad_resize = grad_resize1
                   total_loss = loss
               
                else:
                   grad_resize += grad_resize1
                   total_loss += loss
                
                grad_sum = torch.sum(grad_resize)
                print("ACCUMULATED GRADIENT", i, j, grad_sum)
                
                if j%5 == 0:
                    iteration = jsteps * i + j
                    
                    writer.add_scalar('loss/det_loss', det_loss.detach().cpu().numpy(), iteration)
                    writer.add_scalar('loss/fir_loss', fir_loss.detach().cpu().numpy(), iteration)
                    writer.add_scalar('loss/dist_loss', dist_loss.detach().cpu().numpy(), iteration)
                    writer.add_scalar('loss/satur_loss', satur_loss.detach().cpu().numpy(), iteration)
                    writer.add_scalar('loss/nps_loss', nps_loss.detach().cpu().numpy(), iteration)
                    writer.add_scalar('loss/tv_loss', tv_loss.detach().cpu().numpy(), iteration)
                    writer.add_scalar('misc/epoch', i, iteration)
                    writer.add_scalar('misc/accumulated_gradient', grad_sum, iteration)
                    
#                     if(addweight_transform == "True"):
#                         writer.add_image('misc/patch_fix', patch_fix.squeeze(), iteration) 
#                         writer.add_image('misc/patch_transform', patch_transform.squeeze(), iteration) 
#                         writer.add_image('misc/patch_pers_transform', patch_pers_transform.squeeze(), iteration) 
#                         writer.add_image('misc/patch_resize', patch_resize.squeeze(), iteration) 
#                         writer.add_image('misc/ori_stop2_resize', ori_stop2_resize.squeeze(), iteration) 
#                         writer.add_image('misc/stop_4', stop_4.squeeze(), iteration) 
#                         writer.add_image('misc/adv_stop_img', adv_stop_img.squeeze(), iteration) 
#                         writer.add_image('misc/input1', input1.squeeze(), iteration) 
#                         writer.add_image('misc/input2', input2.squeeze(), iteration) 

#                         writer.add_text('misc/addweight_transform', addweight_transform, iteration) 

                    

            total_loss = total_loss/jsteps
            grad_resize = grad_resize/jsteps  #need to average the gradient?
            

    
            print('\nEpoch, i:', i)
            print('total_loss:',total_loss)

            avg_grad_sum = torch.sum(grad_resize)
            print("AVERAGE GRADIENT, sum:", avg_grad_sum)
            
            end = time.time()
            t = end - start
            print("time taken: ", t)
            
            # add epsilon
            epsilon = 0.05 / (math.floor(i/100) + 1)
            grad_4_patches = grad_resize * map_4_patches
#             print(grad_resize)
#             print(torch.sign(grad_4_patches))
            epsilon_4_patches = epsilon * torch.sign(grad_4_patches) #FGSM attack
            patch_four = patch_four - epsilon_4_patches * map_4_patches
            patch_four = torch.clamp(patch_four, 0, 1)
            
            #random original _stop
            original_stop = get_random_stop_ori(imlist_stop).to(device)
            original_stop = original_stop[0, :, :, :]
            patch_fix = original_stop * map_4_stop + patch_four
            patch_fix = torch.clamp(patch_fix, 0, 1)

            save_every = save_interval
            if i % 5 == 0:
                #det_and_save_img_i(input1, i, output_dir + "adv_img") # Adversarial Example: woth background and adversarial stop sign
                #det_and_save_img_i(input2,i, output_dir + "ori_img")  # Benign Example: img with background and original stop sign
                save_img_i(patch_fix, i, output_dir + "adv_stop/")
                # save_img_i(grad_resize, i, "output/batch/debug/grad_resize/")
                # save_img_i(map_4_patches, i, "output/batch/debug/map_4_patches/")
                # save_img_i(grad_4_patches, i, "output/batch/debug/grad_4_patches/")
                # save_img_i(epsilon_4_patches, i, "output/batch/debug/epsilon_4_patches/")
                # save_img_i(patch_four, i, "output/batch/debug/patch_four/")
                # save_img_i(map_4_stop, i, "output/batch/debug/map_4_stop/")
            if i % 5 == 0:
                iteration = jsteps * (i+1)
                writer.add_scalar('loss/total_loss', total_loss.detach().cpu().numpy(), i)
                writer.add_scalar('misc/learning_rate', epsilon, i)
                writer.add_scalar('misc/avg_grad_sum', avg_grad_sum, i)
                writer.add_image('adv_stop', patch_fix.squeeze(), i)
                writer.add_image('patch', patch_four.squeeze(), i)
                writer.add_image('adv_img', input1.squeeze(), i)
                writer.add_image('grad_resize', grad_resize1.squeeze(), i) 
