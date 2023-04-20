def check_size(model):
  param_size = 0
  buffer_size = 0

  for param in model.parameters():
      param_size += param.nelement() * param.element_size()
  for buffer in model.buffers():
      buffer_size += buffer.nelement() * buffer.element_size()

  size_mb = (param_size + buffer_size) / 1024**2
  print("Size: {:.3f} MB".format(size_mb))
  