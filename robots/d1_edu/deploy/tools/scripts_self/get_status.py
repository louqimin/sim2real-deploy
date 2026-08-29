
import sys, time, math
sys.path.insert(0, "/home/lqm/self-AGI/agibot_D1_Edu-Ultra/lib/zsl-1/x86_64")
import mc_sdk_zsl_1_py

dog = mc_sdk_zsl_1_py.LowLevel()
dog.initRobot("192.168.168.100", 43988, "192.168.168.168")

for _ in range(200):
  if dog.haveMotorData():
      break
  time.sleep(0.05)
else:
  print("没有收到电机数据"); sys.exit(1)

s = dog.getMotorState()
w, x, y, z = dog.getQuaternion()
gx = 2*(w*y - x*z)
gy = -2*(w*x + y*z)
gz = 1 - 2*(w*w + z*z)

print("重力投影 :", round(gx,3), round(gy,3), round(gz,3))
print("角速度   :", [round(v,3) for v in dog.getBodyGyro()])
print("ABAD     :", [round(v,3) for v in s.q_abad])
print("HIP      :", [round(v,3) for v in s.q_hip])
print("KNEE     :", [round(v,3) for v in s.q_knee])


