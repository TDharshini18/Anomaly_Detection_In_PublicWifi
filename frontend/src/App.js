import React, { useState, useCallback, useEffect } from 'react';
import { Layout, Typography, Card, Row, Col, Button, Select, Alert, Spin, Table, Progress, Tag, Space, Divider } from 'antd';
import { UploadOutlined, BugOutlined, SafetyOutlined, WarningOutlined, DownloadOutlined, ReloadOutlined, CheckCircleOutlined } from '@ant-design/icons';
import { useDropzone } from 'react-dropzone';
import Plot from 'react-plotly.js';
import axios from 'axios';
import './App.css';

const { Header, Content } = Layout;
const { Title, Paragraph, Text } = Typography;
const { Option } = Select;

const API_BASE_URL = 'http://localhost:8000';

function App() {
  const [fileData, setFileData] = useState(null);
  const [fileId, setFileId] = useState(null);
  const [columns, setColumns] = useState([]);
  const [columnMapping, setColumnMapping] = useState({});
  const [autoMapping, setAutoMapping] = useState({});
  const [results, setResults] = useState(null);
  const [predictionId, setPredictionId] = useState(null);
  const [loading, setLoading] = useState(false);
  const [step, setStep] = useState(1);
  const [error, setError] = useState(null);
  const [sessionStatus, setSessionStatus] = useState(null);
  const [processingStatus, setProcessingStatus] = useState('');

  const requiredColumns = [
    'signal_strength', 'packet_size', 'frequency', 'connection_duration',
    'data_rate', 'retry_count', 'encryption_type', 'channel'
  ];

  const columnDescriptions = {
    'signal_strength': 'Wi-Fi signal strength (dBm, typically -30 to -100)',
    'packet_size': 'Network packet size (bytes)',
    'frequency': 'Wi-Fi frequency band (2.4, 5.0 GHz)',
    'connection_duration': 'Connection duration (seconds)',
    'data_rate': 'Data transmission rate (Mbps)',
    'retry_count': 'Number of packet retransmissions',
    'encryption_type': 'Security protocol (WPA2, WPA3, Open, etc.)',
    'channel': 'Wi-Fi channel number'
  };

  // Check session status on load
  useEffect(() => {
    checkSessionStatus();
  }, []);

  const checkSessionStatus = async () => {
    try {
      const response = await axios.get(`${API_BASE_URL}/session-status`);
      setSessionStatus(response.data);
    } catch (error) {
      console.error('Error checking session status:', error);
    }
  };

  const onDrop = useCallback(async (acceptedFiles) => {
    const file = acceptedFiles[0];
    if (!file) return;

    setLoading(true);
    setError(null);
    setProcessingStatus('Uploading file...');
    
    try {
      const formData = new FormData();
      formData.append('file', file);

      console.log('Uploading file:', file.name);
      
      const response = await axios.post(`${API_BASE_URL}/upload-file`, formData, {
        headers: { 'Content-Type': 'multipart/form-data' },
        timeout: 60000
      });

      console.log('Upload successful:', response.data);

      setFileData(response.data);
      setFileId(response.data.file_id);
      setColumns(response.data.columns);
      setProcessingStatus('File uploaded successfully!');

      // Auto-detect columns
      setProcessingStatus('Auto-detecting column mappings...');
      const autoDetectResponse = await axios.post(`${API_BASE_URL}/auto-detect-columns`, {
        columns: response.data.columns
      });

      setAutoMapping(autoDetectResponse.data.column_mapping);
      setColumnMapping(autoDetectResponse.data.column_mapping);
      setProcessingStatus('Column mapping completed!');
      setStep(2);
      
      await checkSessionStatus();
    } catch (error) {
      console.error('Error uploading file:', error);
      if (error.code === 'ECONNREFUSED' || error.message.includes('ECONNREFUSED')) {
        setError('Cannot connect to backend server. Make sure the FastAPI server is running on localhost:8000');
      } else if (error.response) {
        setError(`Server error: ${error.response.data.detail || error.response.statusText}`);
      } else {
        setError('Failed to upload file. Please check your connection and try again.');
      }
      setProcessingStatus('');
    }
    setLoading(false);
  }, []);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: {
      'text/csv': ['.csv'],
      'application/json': ['.json'],
      'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet': ['.xlsx'],
      'application/vnd.ms-excel': ['.xls']
    },
    multiple: false
  });

  const handleDemoData = async (type) => {
    setLoading(true);
    setError(null);
    setProcessingStatus(`Loading ${type} demo data...`);
    
    try {
      const response = await axios.get(`${API_BASE_URL}/demo-data/${type}`, {
        timeout: 30000
      });
      
      // Create file-like structure for demo data
      const demoFileData = {
        columns: Object.keys(response.data.data[0]),
        shape: [response.data.data.length, Object.keys(response.data.data[0]).length],
        sample_data: response.data.data,
        total_records: response.data.total_records,
        filename: `demo_${type}_data.json`,
        status: "demo_data_loaded"
      };
      
      setFileData(demoFileData);
      setFileId(null); // No file ID for demo data
      setColumns(Object.keys(response.data.data[0]));
      
      // Direct mapping for demo data
      const directMapping = {};
      Object.keys(response.data.data[0]).forEach(col => {
        directMapping[col] = col;
      });
      setColumnMapping(directMapping);
      setAutoMapping(directMapping);
      setProcessingStatus(`${type} demo data loaded successfully!`);
      setStep(2);
      
      await checkSessionStatus();
    } catch (error) {
      console.error('Error loading demo data:', error);
      setError('Failed to load demo data. Please check your connection and try again.');
      setProcessingStatus('');
    }
    setLoading(false);
  };

  const handleColumnMappingChange = (targetColumn, sourceColumn) => {
    setColumnMapping(prev => ({
      ...prev,
      [targetColumn]: sourceColumn
    }));
  };

  const analyzeData = async () => {
    setLoading(true);
    setError(null);
    setProcessingStatus('Training model and analyzing data...');
    
    try {
      // Send request with proper structure
      const requestData = {
        column_mapping: columnMapping,
        ...(fileId ? { file_id: fileId } : { data: fileData.sample_data })
      };
      
      console.log('Analyzing data with request:', requestData);
      
      const response = await axios.post(`${API_BASE_URL}/predict`, requestData, {
        timeout: 120000 // 2 minutes for training and prediction
      });

      console.log('Analysis complete:', response.data);
      setResults(response.data);
      setPredictionId(response.data.prediction_id);
      setProcessingStatus('Analysis completed successfully!');
      setStep(3);
      
      await checkSessionStatus();
    } catch (error) {
      console.error('Error analyzing data:', error);
      if (error.code === 'ECONNREFUSED' || error.message.includes('ECONNREFUSED')) {
        setError('Cannot connect to backend server. Make sure the FastAPI server is running on localhost:8000');
      } else if (error.response) {
        setError(`Server error: ${error.response.data.detail || error.response.statusText}`);
      } else {
        setError('Failed to analyze data. Please check your connection and try again.');
      }
      setProcessingStatus('');
    }
    setLoading(false);
  };

  const downloadResults = async () => {
    if (!predictionId) {
      setError('No prediction results to download');
      return;
    }

    setLoading(true);
    setProcessingStatus('Preparing download...');
    
    try {
      const response = await axios.get(`${API_BASE_URL}/download-results/${predictionId}`, {
        responseType: 'blob',
        timeout: 30000
      });

      // Create download link
      const url = window.URL.createObjectURL(new Blob([response.data]));
      const link = document.createElement('a');
      link.href = url;
      link.setAttribute('download', `anomaly_results_${predictionId}.xlsx`);
      document.body.appendChild(link);
      link.click();
      link.remove();
      window.URL.revokeObjectURL(url);
      
      setProcessingStatus('Download completed!');
      setTimeout(() => setProcessingStatus(''), 3000);
    } catch (error) {
      console.error('Error downloading results:', error);
      setError('Failed to download results. Please try again.');
      setProcessingStatus('');
    }
    setLoading(false);
  };

  const resetSession = async () => {
    setLoading(true);
    setProcessingStatus('Resetting session...');
    
    try {
      await axios.post(`${API_BASE_URL}/reset-session`);
      
      // Clear all state
      setFileData(null);
      setFileId(null);
      setColumns([]);
      setColumnMapping({});
      setAutoMapping({});
      setResults(null);
      setPredictionId(null);
      setStep(1);
      setError(null);
      setProcessingStatus('Session reset! Ready for new data.');
      
      await checkSessionStatus();
      setTimeout(() => setProcessingStatus(''), 3000);
    } catch (error) {
      console.error('Error resetting session:', error);
      setError('Failed to reset session. Please try again.');
      setProcessingStatus('');
    }
    setLoading(false);
  };

  const renderFileUpload = () => (
    <Card title="📁 Upload Wi-Fi Data" className="upload-card">
      {fileData && (
        <Alert
          message="File uploaded successfully!"
          description={`Found ${fileData.total_records || fileData.shape?.[0] || 'unknown number of'} records in ${fileData.filename || 'the file'}`}
          type="success"
          style={{ marginBottom: 16 }}
          showIcon
          icon={<CheckCircleOutlined />}
        />
      )}
      
      {processingStatus && (
        <Alert
          message="Processing Status"
          description={processingStatus}
          type="info"
          style={{ marginBottom: 16 }}
          showIcon
        />
      )}
      
      <div {...getRootProps()} className={`dropzone ${isDragActive ? 'drag-active' : ''}`}>
        <input {...getInputProps()} />
        <div style={{ textAlign: 'center', padding: '40px', position: 'relative', zIndex: 1 }}>
          <UploadOutlined className="metrics-icon" style={{ color: '#667eea' }} />
          {isDragActive ? (
            <div>
              <p style={{ fontSize: '18px', fontWeight: '500', marginTop: '16px' }}>
                Drop the file here...
              </p>
            </div>
          ) : (
            <div>
              <p style={{ fontSize: '18px', fontWeight: '500', marginTop: '16px' }}>
                Drag & drop a file here, or click to select
              </p>
              <p style={{ color: '#666', fontSize: '14px' }}>
                Supports CSV, JSON, Excel files (up to 100MB)
              </p>
            </div>
          )}
        </div>
      </div>
      
      <Row gutter={16} style={{ marginTop: 24 }}>
        <Col span={12}>
          <Button 
            type="default" 
            block 
            size="large"
            onClick={() => handleDemoData('normal')}
            icon={<SafetyOutlined />}
            style={{ height: '48px' }}
          >
            Try Demo Data (Normal)
          </Button>
        </Col>
        <Col span={12}>
          <Button 
            type="default" 
            block 
            size="large"
            onClick={() => handleDemoData('suspicious')}
            icon={<WarningOutlined />}
            style={{ height: '48px' }}
          >
            Try Demo Data (Suspicious)
          </Button>
        </Col>
      </Row>
    </Card>
  );

  const renderColumnMapping = () => (
    <Card title="🔧 Column Mapping" className="mapping-card">
      {Object.keys(autoMapping).length > 0 && (
        <Alert
          message={`Auto-detected ${Object.keys(autoMapping).length} column mappings`}
          type="success"
          style={{ marginBottom: 16 }}
          showIcon
        />
      )}
      
      <div className="column-mapping-section">
        <Row gutter={[16, 16]}>
          {requiredColumns.map(reqCol => (
            <Col span={12} key={reqCol}>
              <div className="column-item">
                <div style={{ marginBottom: 12 }}>
                  <Text strong style={{ fontSize: '16px', color: '#333' }}>
                    {reqCol.replace(/_/g, ' ').toUpperCase()}
                  </Text>
                  <br />
                  <Text type="secondary" style={{ fontSize: '12px', lineHeight: '1.4' }}>
                    {columnDescriptions[reqCol]}
                  </Text>
                </div>
                <Select
                  style={{ width: '100%' }}
                  placeholder="Select column"
                  value={columnMapping[reqCol]}
                  onChange={(value) => handleColumnMappingChange(reqCol, value)}
                  size="large"
                >
                  <Option value="">None</Option>
                  {columns.map(col => (
                    <Option key={col} value={col}>{col}</Option>
                  ))}
                </Select>
              </div>
            </Col>
          ))}
        </Row>
      </div>

      {processingStatus && (
        <Alert
          message="Processing Status"
          description={processingStatus}
          type="info"
          style={{ marginTop: 16 }}
          showIcon
        />
      )}

      <div style={{ marginTop: 32, textAlign: 'center' }}>
        <Button 
          type="primary" 
          size="large"
          onClick={analyzeData}
          disabled={Object.keys(columnMapping).filter(key => columnMapping[key]).length === 0}
          icon={<BugOutlined />}
          style={{ height: '48px', fontSize: '16px', padding: '0 32px' }}
        >
          Train Model & Analyze Data
        </Button>
      </div>
    </Card>
  );

  const renderResults = () => {
    if (!results) return null;

    const getStatusColor = (isAnomalous) => isAnomalous ? '#ff4d4f' : '#52c41a';
    const getStatusText = (isAnomalous) => isAnomalous ? 'SUSPICIOUS' : 'SAFE';

    return (
      <div>
        <Alert
          message={`Analysis Complete: Processed ${results.total_records} records`}
          description={`Prediction ID: ${results.prediction_id} | Completed at: ${new Date(results.timestamp).toLocaleString()}`}
          type="success"
          style={{ marginBottom: 24 }}
          showIcon
        />
        
        <Row gutter={[24, 24]} style={{ marginBottom: 32 }}>
          <Col span={6}>
            <div className={`metrics-card ${results.overall_anomaly ? 'suspicious' : 'safe'}`}>
              {results.overall_anomaly ? (
                <WarningOutlined className="metrics-icon" style={{ color: '#ff4d4f' }} />
              ) : (
                <SafetyOutlined className="metrics-icon" style={{ color: '#52c41a' }} />
              )}
              <div className="metrics-title" style={{ color: getStatusColor(results.overall_anomaly) }}>
                {getStatusText(results.overall_anomaly)}
              </div>
              <p className="metrics-subtitle">Network Status</p>
            </div>
          </Col>
          <Col span={6}>
            <div className="metrics-card">
              <div className="metrics-value" style={{ color: '#667eea' }}>
                {results.avg_anomaly_score.toFixed(2)}
              </div>
              <div className="metrics-title">Anomaly Score</div>
              <p className="metrics-subtitle">Threshold: 1.0</p>
            </div>
          </Col>
          <Col span={6}>
            <div className="metrics-card">
              <div className="metrics-value" style={{ color: '#764ba2' }}>
                {results.anomaly_percentage.toFixed(1)}%
              </div>
              <div className="metrics-title">Anomalous Records</div>
              <p className="metrics-subtitle">
                {results.is_anomaly.filter(Boolean).length} / {results.total_records} records
              </p>
            </div>
          </Col>
          <Col span={6}>
            <div className="metrics-card">
              <div className="metrics-value" style={{ color: '#52c41a' }}>
                {results.total_records}
              </div>
              <div className="metrics-title">Total Records</div>
              <p className="metrics-subtitle">Processed successfully</p>
            </div>
          </Col>
        </Row>

        <Card title="📊 Anomaly Score Distribution" className="plot-card" style={{ marginBottom: 24 }}>
          <div className="plot-container">
            <Plot
              data={[
                {
                  x: results.anomaly_scores,
                  type: 'histogram',
                  nbinsx: 30,
                  marker: { 
                    color: '#667eea',
                    opacity: 0.7,
                    line: { color: '#764ba2', width: 1 }
                  }
                }
              ]}
              layout={{
                title: {
                  text: 'Distribution of Anomaly Scores',
                  font: { size: 18, color: '#333' }
                },
                xaxis: { 
                  title: 'Anomaly Score',
                  gridcolor: '#f0f0f0'
                },
                yaxis: { 
                  title: 'Frequency',
                  gridcolor: '#f0f0f0'
                },
                plot_bgcolor: 'white',
                paper_bgcolor: 'white',
                shapes: [{
                  type: 'line',
                  x0: 1,
                  y0: 0,
                  x1: 1,
                  y1: 1,
                  yref: 'paper',
                  line: { color: '#ff4d4f', width: 3, dash: 'dash' }
                }],
                annotations: [{
                  x: 1,
                  y: 1,
                  yref: 'paper',
                  text: 'Threshold',
                  showarrow: true,
                  arrowhead: 2,
                  arrowcolor: '#ff4d4f',
                  font: { color: '#ff4d4f', size: 12 }
                }]
              }}
              style={{ width: '100%', height: '400px' }}
              config={{ displayModeBar: false }}
            />
          </div>
        </Card>

        <Card title="📋 Prediction Results Summary" style={{ marginBottom: 24 }}>
          <Table
            dataSource={results.anomaly_scores.slice(0, 100).map((score, index) => ({
              key: index,
              record_id: index + 1,
              anomaly_score: score.toFixed(3),
              is_anomalous: results.is_anomaly[index],
              reconstruction_error: results.reconstruction_errors[index].toFixed(6),
              status: results.is_anomaly[index] ? 'ANOMALOUS' : 'NORMAL'
            }))}
            columns={[
              {
                title: 'Record ID',
                dataIndex: 'record_id',
                key: 'record_id',
                width: 100
              },
              {
                title: 'Anomaly Score',
                dataIndex: 'anomaly_score',
                key: 'anomaly_score',
                render: (score) => (
                  <Tag color={parseFloat(score) > 1 ? 'red' : 'green'}>
                    {score}
                  </Tag>
                ),
                sorter: (a, b) => parseFloat(a.anomaly_score) - parseFloat(b.anomaly_score)
              },
              {
                title: 'Status',
                dataIndex: 'status',
                key: 'status',
                render: (status, record) => (
                  <Tag color={record.is_anomalous ? 'red' : 'green'}>
                    {status}
                  </Tag>
                ),
                filters: [
                  { text: 'Normal', value: 'NORMAL' },
                  { text: 'Anomalous', value: 'ANOMALOUS' }
                ],
                onFilter: (value, record) => record.status === value
              },
              {
                title: 'Reconstruction Error',
                dataIndex: 'reconstruction_error',
                key: 'reconstruction_error',
                sorter: (a, b) => parseFloat(a.reconstruction_error) - parseFloat(b.reconstruction_error)
              }
            ]}
            pagination={{ 
              pageSize: 20,
              showTotal: (total, range) => `${range[0]}-${range[1]} of ${total} items (Showing first 100 records)`
            }}
            rowClassName={(record) => record.is_anomalous ? 'anomaly-row' : 'normal-row'}
            scroll={{ x: 800 }}
          />
          
          {results.total_records > 100 && (
            <Alert
              message={`Showing first 100 records only. Download Excel file to see all ${results.total_records} results.`}
              type="info"
              style={{ marginTop: 16 }}
              showIcon
            />
          )}
        </Card>

        <Card title="💡 Actions & Recommendations">
          <Space direction="vertical" style={{ width: '100%' }}>
            <div style={{ textAlign: 'center', marginBottom: 16 }}>
              <Space size="large">
                <Button 
                  type="primary"
                  size="large"
                  icon={<DownloadOutlined />}
                  onClick={downloadResults}
                  style={{ height: '48px', fontSize: '16px' }}
                >
                  Download Results (Excel)
                </Button>
                
                <Button 
                  size="large"
                  icon={<ReloadOutlined />}
                  onClick={resetSession}
                  style={{ height: '48px', fontSize: '16px' }}
                >
                  Start New Analysis
                </Button>
              </Space>
            </div>
            
            <Divider />
            
            {results.overall_anomaly ? (
              <Alert
                message="⚠️ Suspicious Activity Detected!"
                description={
                  <div style={{ marginTop: 12 }}>
                    <ul style={{ margin: 0, paddingLeft: 20 }}>
                      <li>Monitor this network closely for unusual patterns</li>
                      <li>Check for signs of evil twin attacks or data harvesting</li>
                      <li>Verify the network's legitimacy and authorization</li>
                      <li>Consider implementing additional security measures</li>
                      <li>Alert security team if patterns persist</li>
                      <li>Download the detailed results for further analysis</li>
                    </ul>
                  </div>
                }
                type="warning"
                showIcon
                style={{ fontSize: '16px' }}
              />
            ) : (
              <Alert
                message="✅ Network Appears Safe"
                description={
                  <div style={{ marginTop: 12 }}>
                    <ul style={{ margin: 0, paddingLeft: 20 }}>
                      <li>Traffic patterns are within normal operational range</li>
                      <li>Continue regular monitoring and security practices</li>
                      <li>Maintain updated security protocols and software</li>
                      <li>Schedule periodic security assessments</li>
                      <li>Download results for record keeping</li>
                    </ul>
                  </div>
                }
                type="success"
                showIcon
                style={{ fontSize: '16px' }}
              />
            )}
          </Space>
        </Card>
      </div>
    );
  };

  return (
    <Layout style={{ minHeight: '100vh' }}>
      <Header style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
        <Title level={2} style={{ margin: '16px 0', color: 'white' }}>
          🛡️ Wi-Fi Network Anomaly Detection
        </Title>
        
        {sessionStatus && (
          <Space style={{ color: 'white' }}>
            <Text style={{ color: 'white' }}>
              Files: {sessionStatus.uploaded_files} | Predictions: {sessionStatus.predictions}
            </Text>
          </Space>
        )}
      </Header>
      
      <Content style={{ padding: '24px' }}>
        <div className="main-container" style={{ maxWidth: '1200px', margin: '0 auto' }}>
          <div style={{ textAlign: 'center', marginBottom: '32px' }}>
            <Title level={3} style={{ color: '#333', marginBottom: '8px' }}>
              Advanced Network Security Analysis
            </Title>
            <Paragraph style={{ fontSize: '16px', color: '#666', maxWidth: '600px', margin: '0 auto' }}>
              Upload Wi-Fi traffic data to detect potentially malicious or anomalous networks using machine learning
            </Paragraph>
          </div>

          <div style={{ marginBottom: '32px' }}>
            <Progress 
              percent={step * 33.33} 
              steps={3} 
              strokeColor={{
                '0%': '#667eea',
                '100%': '#764ba2',
              }}
              trailColor="rgba(255, 255, 255, 0.3)"
              format={() => `Step ${step} of 3`}
              style={{ fontSize: '14px', fontWeight: '500' }}
            />
          </div>

          {error && (
            <Alert
              message="Error"
              description={error}
              type="error"
              closable
              onClose={() => setError(null)}
              style={{ marginBottom: 24 }}
              showIcon
            />
          )}

          <Spin spinning={loading} size="large" tip={processingStatus || "Processing..."}>
            {step === 1 && renderFileUpload()}
            {step === 2 && renderColumnMapping()}
            {step === 3 && renderResults()}
          </Spin>

          {step > 1 && step < 3 && (
            <div style={{ textAlign: 'center', marginTop: 32 }}>
              <Button 
                size="large"
                onClick={() => setStep(step - 1)}
                style={{ marginRight: 16, height: '40px' }}
              >
                ← Back
              </Button>
            </div>
          )}
        </div>
      </Content>
    </Layout>
  );
}

export default App;