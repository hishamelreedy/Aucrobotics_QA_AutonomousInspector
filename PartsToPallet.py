# Type help("robolink") or help("robodk") for more information
# Press F5 to run the script
# Note: you do not need to keep a copy of this file, your python script is saved with the station
from robolink import *    # API to communicate with RoboDK
from robodk import *      # basic matrix operations

# Use RoboDK API as RL
RDK = Robolink()

# define default approach distance
APPROACH = 100
SENSOR_VARIABLE = 'SENSOR' # station variable
PART_KEYWORD = 'Part' # parts to look for (camera simulation)

# gather robot, tool and reference frames from the station
robot               = RDK.Item('UR10 B', ITEM_TYPE_ROBOT)
tool                = RDK.Item('GripperB', ITEM_TYPE_TOOL)
frame_pallet        = RDK.Item('PalletB', ITEM_TYPE_FRAME)
frame_conv          = RDK.Item('ConveyorReference', ITEM_TYPE_FRAME)
frame_conv_moving   = RDK.Item('MovingRef', ITEM_TYPE_FRAME)

# gather targets
target_pallet_safe = RDK.Item('PalletApproachB', ITEM_TYPE_TARGET)
target_conv_safe = RDK.Item('ConvApproachB', ITEM_TYPE_TARGET)
target_conv = RDK.Item('Get Conveyor', ITEM_TYPE_TARGET)
target_inspect = RDK.Item('InspectPartB', ITEM_TYPE_TARGET)

# get variable parameters
SIZE_BOX = RDK.getParam('SizeBox')
SIZE_PALLET = RDK.getParam('SizePallet')
SIZE_BOX_XYZ = [float(x.replace(' ','')) for x in SIZE_BOX.split(',')]
SIZE_PALLET_XYZ = [float(x.replace(' ','')) for x in SIZE_PALLET.split(',')]
SIZE_BOX_Z = SIZE_BOX_XYZ[2] # the height of the boxes is important to take into account when approaching the positions

camera_ref_conv = target_conv.PoseAbs()

#----------------- camera simulation
# Get all object names to
all_objects = RDK.ItemList(ITEM_TYPE_OBJECT, True)

# Get object items in a list (faster) and filter by keyword
check_objects = []
for i in range(len(all_objects)):
    if all_objects[i].count(PART_KEYWORD) > 0:
        check_objects.append(RDK.Item(all_objects[i]))
        
# Make sure that there is at least one part that we are expecting
if len(check_objects) == 0:
    raise Exception('No parts to check for. Name at least one part with the name: %s.' % PICKABLE_OBJECTS_KEYWORD)

# simulates the behavior of the camera (returns TX, TY and RZ)
def WaitPartCamera():
    """Simulate camera detection"""
    if RDK.RunMode() == RUNMODE_SIMULATE:
        # Simulate the camera by waiting for an object to be detected
        while True:
            for part in check_objects:
                # calculate the position of the part with respect to the target
                part_pose = invH(camera_ref_conv) * part.PoseAbs()
                tx,ty,tz,rx,ry,rz = pose_2_xyzrpw(part_pose)
                rz = rz * pi/180.0 # Convert degrees to radians
                if abs(tx) < 400 and ty < 50 and abs(tz) < 400:
                    print('Part detected: TX,TY,TZ=%.1f,%.1f,%.1f' % (tx,ty,rz))
                    return tx, ty, rz
            pause(0.005)
    else:
        RDK.RunProgram('WaitPartCamera')
    return 0,0,0

#-----------------------------

def box_calc(size_xyz, pallet_xyz):
    """Calculates a list of points to store parts in a pallet"""
    [size_x, size_y, size_z] = size_xyz
    [pallet_x, pallet_y, pallet_z] = pallet_xyz    
    xyz_list = []
    for h in range(int(pallet_z)):
        for j in range(int(pallet_y)):
            for i in range(int(pallet_x)):
                xyz_list = xyz_list + [[(i+0.5)*size_x, (j+0.5)*size_y, (h+0.5)*size_z]]
    return xyz_list

def TCP_On(toolitem):
    """Attaches the closest object to the toolitem Htool pose,
    It will also output appropriate function calls on the generated robot program (call to TCP_On)"""
    toolitem.AttachClosest()
    toolitem.RDK().RunMessage('Set air valve on')
    toolitem.RDK().RunProgram('TCP_On()');
        
def TCP_Off(toolitem, itemleave=0):
    """Detaches the closest object attached to the toolitem Htool pose,
    It will also output appropriate function calls on the generated robot program (call to TCP_Off)"""
    #toolitem.DetachClosest(itemleave)
    toolitem.DetachAll(itemleave)
    toolitem.RDK().RunMessage('Set air valve off')
    toolitem.RDK().RunProgram('TCP_Off()');

# calculate an array of positions to get/store the parts
parts_positions = box_calc(SIZE_BOX_XYZ, SIZE_PALLET_XYZ)

# Calculate a new TCP that takes into account the size of the part (targets are set to the center of the box)
tool_xyzrpw = tool.PoseTool()*transl(0,0,SIZE_BOX_Z/2)
tool_tcp = robot.AddTool(tool_xyzrpw, 'TCP B')

# ---------------------------------------------------------------------------------
# -------------------------- PROGRAM START ----------------------------------------
robot.setPoseTool(tool_tcp)
nparts = len(parts_positions)
i = 0
while i < nparts:
    # ----------- take a part from the convegor ------
    # approach to the conveyor
    robot.setPoseFrame(frame_conv)
    robot.MoveJ(target_inspect)

    #------------------------- camera simulation
    CAM_TX, CAM_TY, CAM_RZ = WaitPartCamera()
    # Adjust position along movement according to conveyor speed
    CAM_TY = CAM_TY - 50
    # Lazy move: the part is symmetrical so there is no need to turn more than +/-90 deg around the Z axis. If so, add as many half turns as required
    if CAM_RZ > pi/2:
        CAM_RZ = CAM_RZ - pi
    elif CAM_RZ < -pi/2:
        CAM_RZ = CAM_RZ + pi
    target_conv_pose = target_conv.Pose()*transl(CAM_TX,CAM_TY,-SIZE_BOX_Z/2)*rotz(CAM_RZ)
    target_conv_app = target_conv_pose*transl(0,0,-APPROACH)
    #-------------------------------------------------
    
    robot.MoveL(target_conv_pose)
    TCP_On(tool) # detach an place the object in the moving reference frame of the conveyor
    robot.MoveL(target_conv_app)
    
    if len(tool.Childs()) == 0:
        # Detect invalid pick
        continue
    
    robot.MoveJ(target_conv_safe)

    # ----------- place the part on the pallet ------
    # get the xyz position of part i
    robot.setPoseFrame(frame_pallet)
    part_position_i = parts_positions[i]
    target_i = transl(part_position_i)*rotx(pi)
    target_i_app = target_i * transl(0,0,-(APPROACH+SIZE_BOX_Z))    
    # approach to the pallet
    robot.MoveJ(target_pallet_safe)
    # get the box i
    robot.MoveJ(target_i_app)
    robot.MoveL(target_i)
    TCP_Off(tool, frame_pallet) # attach the closest part
    robot.MoveL(target_i_app)
    robot.MoveJ(target_pallet_safe)
    
    i = i + 1
    
    
