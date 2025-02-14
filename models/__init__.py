'''
The purpose of models/__init__.py is to import different model classes and organize them into the ALL_MODELS dictionary. 
This allows for dynamic model selection based on the configuration provided in the YAML file (e.g., vggish_finetune.yaml).
'''

from .base import AspedModel
# from .teacher_student import TeacherStudentModel
# from .video_distillation import VideoDistillationModel

# A dictionary that stores references to three different model classes:
ALL_MODELS = dict(
    base=AspedModel,
    # teacher_student=TeacherStudentModel,
    # video=VideoDistillationModel
)

'''
base: Refers to the AspedModel, used as a baseline model.
teacher_student: Refers to the TeacherStudentModel, used for knowledge distillation (i.e., transferring knowledge from a large, complex model to a smaller one).
video: Refers to the VideoDistillationModel, combining audio and video data for multimodal distillation.
'''