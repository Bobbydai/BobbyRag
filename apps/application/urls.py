from django.urls import path

from . import views

app_name = "application"
urlpatterns = [
    path('application', views.Application.as_view(), name="application"),
    path('application/profile', views.Application.Profile.as_view(), name='application/profile'),
    path('application/embed', views.Application.Embed.as_view()),
    path('application/authentication', views.Application.Authentication.as_view()),
    path('application/<str:application_id>/publish', views.Application.Publish.as_view()),
    path('application/<str:application_id>/edit_icon', views.Application.EditIcon.as_view()),
    path('application/<str:application_id>/statistics/customer_count',
         views.ApplicationStatistics.CustomerCount.as_view()),
    path('application/<str:application_id>/statistics/customer_count_trend',
         views.ApplicationStatistics.CustomerCountTrend.as_view()),
    path('application/<str:application_id>/statistics/chat_record_aggregate',
         views.ApplicationStatistics.ChatRecordAggregate.as_view()),
    path('application/<str:application_id>/statistics/chat_record_aggregate_trend',
         views.ApplicationStatistics.ChatRecordAggregateTrend.as_view()),
    path('application/<str:application_id>/model', views.Application.Model.as_view()),
    path('application/<str:application_id>/function_lib', views.Application.FunctionLib.as_view()),
    path('application/<str:application_id>/function_lib/<str:function_lib_id>',
         views.Application.FunctionLib.Operate.as_view()),
    path('application/<str:application_id>/application', views.Application.Application.as_view()),
    path('application/<str:application_id>/model_params_form/<str:model_id>',
         views.Application.ModelParamsForm.as_view()),
    path('application/<str:application_id>/hit_test', views.Application.HitTest.as_view()),
    path('application/<str:application_id>/api_key', views.Application.ApplicationKey.as_view()),
    path("application/<str:application_id>/api_key/<str:api_key_id>",
         views.Application.ApplicationKey.Operate.as_view()),
    path('application/<str:application_id>', views.Application.Operate.as_view(), name='application/operate'),
    path('application/<str:application_id>/list_dataset', views.Application.ListApplicationDataSet.as_view(),
         name='application/dataset'),
    path('application/<str:application_id>/access_token', views.Application.AccessToken.as_view(),
         name='application/access_token'),
    path('application/<int:current_page>/<int:page_size>', views.Application.Page.as_view(), name='application_page'),
    path('application/<str:application_id>/chat/open', views.ChatView.Open.as_view(), name='application/open'),
    path("application/chat/open", views.ChatView.OpenTemp.as_view()),
    path("application/chat_workflow/open", views.ChatView.OpenWorkFlowTemp.as_view()),
    path("application/<str:application_id>/chat/client/<int:current_page>/<int:page_size>",
         views.ChatView.ClientChatHistoryPage.as_view()),
    path("application/<str:application_id>/chat/client/<chat_id>",
         views.ChatView.ClientChatHistoryPage.Operate.as_view()),
    path('application/<str:application_id>/chat/export', views.ChatView.Export.as_view(), name='export'),
    path('application/<str:application_id>/chat/completions', views.Openai.as_view(),
         name='application/chat_completions'),
    path('application/<str:application_id>/chat', views.ChatView.as_view(), name='chats'),
    path('application/<str:application_id>/chat/<int:current_page>/<int:page_size>', views.ChatView.Page.as_view()),
    path('application/<str:application_id>/chat/<chat_id>', views.ChatView.Operate.as_view()),
    path('application/<str:application_id>/chat/<chat_id>/chat_record/', views.ChatView.ChatRecord.as_view()),
    path('application/<str:application_id>/chat/<chat_id>/upload_file', views.ChatView.UploadFile.as_view()),
    path('application/<str:application_id>/chat/<chat_id>/chat_record/<int:current_page>/<int:page_size>',
         views.ChatView.ChatRecord.Page.as_view()),
    path('application/<str:application_id>/chat/<chat_id>/chat_record/<chat_record_id>',
         views.ChatView.ChatRecord.Operate.as_view()),
    path('application/<str:application_id>/chat/<chat_id>/chat_record/<str:chat_record_id>/vote',
         views.ChatView.ChatRecord.Vote.as_view(),
         name=''),
    path(
        'application/<str:application_id>/chat/<chat_id>/chat_record/<str:chat_record_id>/dataset/<str:dataset_id>/document_id/<str:document_id>/improve',
        views.ChatView.ChatRecord.Improve.as_view(),
        name=''),
    path('application/<str:application_id>/chat/<chat_id>/chat_record/<str:chat_record_id>/improve',
         views.ChatView.ChatRecord.ChatRecordImprove.as_view()),
    path('application/chat_message/<str:chat_id>', views.ChatView.Message.as_view(), name='application/message'),
    path(
        'application/<str:application_id>/chat/<chat_id>/chat_record/<str:chat_record_id>/dataset/<str:dataset_id>/document_id/<str:document_id>/improve/<str:paragraph_id>',
        views.ChatView.ChatRecord.Improve.Operate.as_view(),
        name=''),
    path('application/<str:application_id>/speech_to_text', views.Application.SpeechToText.as_view(),
         name='application/audio'),
    path('application/<str:application_id>/text_to_speech', views.Application.TextToSpeech.as_view(),
         name='application/audio'),
    path('application/<str:application_id>/work_flow_version', views.ApplicationVersionView.as_view()),
    path('application/<str:application_id>/work_flow_version/<int:current_page>/<int:page_size>',
         views.ApplicationVersionView.Page.as_view()),
    path('application/<str:application_id>/work_flow_version/<str:work_flow_version_id>',
         views.ApplicationVersionView.Operate.as_view()),
    path('application/<str:application_id>/play_demo_text', views.Application.PlayDemoText.as_view(),
         name='application/audio'),
    path('application/<str:application_id>/content_generate', views.Application.ContentGenerate.as_view(),
         name='application/content_generate'),
    path('application/<str:application_id>/choose_comment', views.Application.ChooseComment.as_view(),
         name='application/choose_comment')
]
