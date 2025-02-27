// Wait for the DOM to be fully loaded
document.addEventListener('DOMContentLoaded', function() {
  // Enhance copy to clipboard functionality
  enhanceCopyToClipboard();
  
  // Enhance error display
  enhanceErrorDisplay();
  
  // Enhance table display
  enhanceTableDisplay();
  
  // Add responsive behavior
  addResponsiveBehavior();
  
  // Enhance optimization results
  enhanceOptimizationResults();
});

// Enhance copy to clipboard functionality
function enhanceCopyToClipboard() {
  const originalCopyToClipboard = window.copyToClipboard;
  if (originalCopyToClipboard) {
      window.copyToClipboard = function(text) {
          originalCopyToClipboard.apply(this, arguments);
          
          // Show a temporary tooltip
          const tooltip = document.createElement('div');
          tooltip.textContent = 'Copied!';
          tooltip.style.position = 'fixed';
          tooltip.style.backgroundColor = 'rgba(0, 0, 0, 0.7)';
          tooltip.style.color = 'white';
          tooltip.style.padding = '5px 10px';
          tooltip.style.borderRadius = '4px';
          tooltip.style.zIndex = '1000';
          tooltip.style.top = (event.clientY - 30) + 'px';
          tooltip.style.left = event.clientX + 'px';
          tooltip.style.opacity = '0';
          tooltip.style.transition = 'opacity 0.3s ease';
          
          document.body.appendChild(tooltip);
          
          setTimeout(() => {
              tooltip.style.opacity = '1';
              
              setTimeout(() => {
                  tooltip.style.opacity = '0';
                  setTimeout(() => {
                      document.body.removeChild(tooltip);
                  }, 300);
              }, 1500);
          }, 10);
      };
  }
}

// Enhance error display
function enhanceErrorDisplay() {
  const originalShowError = window.showError;
  if (originalShowError) {
      window.showError = function(message) {
          originalShowError.apply(this, arguments);
          
          const errorElement = document.getElementById('error');
          if (errorElement) {
              if (message) {
                  errorElement.style.display = 'block';
                  errorElement.classList.add('fade-in');
              } else {
                  errorElement.style.display = 'none';
                  errorElement.classList.remove('fade-in');
              }
          }
      };
  }
}

// Enhance table display
function enhanceTableDisplay() {
  // Check for DataTable initialization
  const checkDataTable = setInterval(() => {
      const dataTable = document.querySelector('.dataTable');
      if (dataTable) {
          clearInterval(checkDataTable);
          
          // Add responsive classes
          dataTable.classList.add('table-responsive');
          
          // Add hover effect to rows
          const rows = dataTable.querySelectorAll('tbody tr');
          rows.forEach(row => {
              row.addEventListener('mouseenter', function() {
                  this.style.backgroundColor = 'rgba(52, 152, 219, 0.05)';
              });
              
              row.addEventListener('mouseleave', function() {
                  this.style.backgroundColor = '';
              });
          });
      }
  }, 1000);
}

// Add responsive behavior
function addResponsiveBehavior() {
  // Adjust viewer height based on screen size
  function adjustViewerHeight() {
      const viewers = ['viewer', 'viewer3D'];
      const width = window.innerWidth;
      
      viewers.forEach(viewerId => {
          const viewer = document.getElementById(viewerId);
          if (!viewer) return;
          
          if (width < 576) {
              viewer.style.height = '250px';
          } else if (width < 768) {
              viewer.style.height = '300px';
          } else if (width < 992) {
              viewer.style.height = '350px';
          } else {
              viewer.style.height = '400px';
          }
      });
  }
  
  // Initial adjustment
  adjustViewerHeight();
  
  // Adjust on window resize
  window.addEventListener('resize', adjustViewerHeight);
}

// Enhance optimization results
function enhanceOptimizationResults() {
  const originalOptimizeLead = window.optimizeLead;
  if (originalOptimizeLead) {
      window.optimizeLead = function(leadSmiles, leadIc50) {
          originalOptimizeLead.apply(this, arguments);
          
          // Scroll to optimization results after they're loaded
          const checkResults = setInterval(() => {
              const results = document.getElementById('optimizationResults');
              if (results && results.innerHTML.trim() !== '') {
                  clearInterval(checkResults);
                  
                  setTimeout(() => {
                      results.scrollIntoView({ behavior: 'smooth', block: 'start' });
                  }, 500);
              }
          }, 500);
      };
  }
}