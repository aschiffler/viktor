{{/*
Expand the name of the chart.
*/}}
{{- define "lightrag-chart.name" -}}
{{- default .Chart.Name .Values.nameOverride | trunc 63 | trimSuffix "-" -}}
{{- end -}}

{{/*
Create a default fully qualified app name.
We truncate at 63 chars because some Kubernetes name fields are limited to this (by the DNS naming spec).
If it's not provided, it will be the chart name.
*/}}
{{- define "lightrag-chart.fullname" -}}
{{- if .Values.fullnameOverride -}}
{{- .Values.fullnameOverride | trunc 63 | trimSuffix "-" -}}
{{- else -}}
{{- $name := default .Chart.Name .Values.nameOverride -}}
{{- if contains $name .Release.Name -}}
{{- .Release.Name | trunc 63 | trimSuffix "-" -}}
{{- else -}}
{{- printf "%s-%s" .Release.Name $name | trunc 63 | trimSuffix "-" -}}
{{- end -}}
{{- end -}}
{{- end -}}

{{/*
Create chart name and version as part of the labels
*/}}
{{- define "lightrag-chart.labels" -}}
helm.sh/chart: {{ include "lightrag-chart.name" . }}-{{ .Chart.Version }}
{{ include "lightrag-chart.selectorLabels" . }}
{{- if .Chart.AppVersion }}
app.kubernetes.io/version: {{ .Chart.AppVersion | quote }}
{{- end }}
app.kubernetes.io/managed-by: {{ .Release.Service }}
{{- end -}}

{{/*
Selector labels
*/}}
{{- define "lightrag-chart.selectorLabels" -}}
app.kubernetes.io/name: {{ include "lightrag-chart.name" . }}
app.kubernetes.io/instance: {{ .Release.Name }}
{{- end -}}

{{/*
Create the name for the lightrag deployment.
*/}}
{{- define "lightrag.fullname" -}}
{{- printf "%s-%s" (include "lightrag-chart.fullname" .) .Values.lightrag.name | trunc 63 | trimSuffix "-" -}}
{{- end -}}

{{/*
Create the name for the neo4j deployment.
*/}}
{{- define "neo4j.fullname" -}}
{{- printf "%s-%s" (include "lightrag-chart.fullname" .) .Values.neo4j.name | trunc 63 | trimSuffix "-" -}}
{{- end -}}